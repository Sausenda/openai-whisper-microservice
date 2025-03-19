import io
import os
import whisper
import operator
from whisper.utils import WriteTXT, ResultWriter, WriteSRT, WriteTSV, WriteVTT, WriteJSON
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile, HTTPException, BackgroundTasks, Depends, Form, APIRouter, Request

from src.database import get_db, Query
from src.helpers import download, load_audio

# Create FastAPI router
router = APIRouter(
    tags=["Model"],
)


@router.get("/models-available", status_code=200)
async def get_available_models():
    """
    Return a list of all available Whisper ASR models.
    """
    models = whisper.available_models()
    return models


@router.get("/models-downloading", status_code=200)
async def get_downloading_models(db = Depends(get_db)):
    """
    Return a list of all downloading Whisper ASR models.
    """
    models = db.search(Query().downloaded == False)
    models = [model["name"] for model in models]
    return models


@router.get("/models-downloaded", status_code=200)
async def get_downloaded_models(db = Depends(get_db)):
    """
    Return a list of all downloaded Whisper ASR models.
    """
    models = db.search(Query().downloaded == True)
    models = [model["name"] for model in models]
    return models


@router.post("/model", status_code=201)
async def download_model(model_name: str, background_tasks: BackgroundTasks, db = Depends(get_db)):
    """
    Download a Whisper ASR model using background task.
    """
    models_downloaded = await get_downloaded_models(db)
    if model_name in models_downloaded:
        raise HTTPException(status_code=409, detail="Model already exist")
    
    models_downloading = await get_downloading_models(db)
    if model_name in models_downloading:
        raise HTTPException(status_code=400, detail="Model not downloaded yet")
    
    models_available = await get_available_models()
    if model_name not in models_available:
        raise HTTPException(status_code=400, detail="Invalid model")
    
    background_tasks.add_task(download, model_name, db)
    return {"detail": "Model is being downloaded"}


@router.get("/model/{model_name}", status_code=200)
async def get_model(model_name: str, db = Depends(get_db)):
    """
    Return a Whisper ASR model.
    """
    result = db.search(Query().name == model_name)
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    elif result[0]["downloaded"] == False:
        raise HTTPException(status_code=400, detail="Model not downloaded yet")
    else:
        model = whisper.load_model(model_name)
        return model


@router.delete("/model/{model_name}", status_code=200)
async def delete_model(model_name: str, db = Depends(get_db)):
    """
    Delete a downloaded Whisper ASR model.
    """
    result = db.search(Query().name == model_name)
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    elif result[0]["downloaded"] == False:
        raise HTTPException(status_code=400, detail="Model not downloaded yet")
    else:
        os.remove(f"../root/.cache/whisper/{model_name}.pt")
        db.remove(Query().name == model_name)   
        return {"detail":"Model was deleted"}


@router.post("/model/{model_name}/language", status_code=200)
async def detect_language(model_name: str, file: UploadFile = File(...), db = Depends(get_db)):
    """
    Return a sorted list of all detected languages by their score.
    """
    if ".en" in model_name:
        raise HTTPException(status_code=400, detail="Model not multilingual")

    result = db.search(Query().name == model_name)
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    elif result[0]["downloaded"] == False:
        raise HTTPException(status_code=400, detail="Model not downloaded yet")
        
    model = whisper.load_model(model_name)

    audio = load_audio(file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    probs_sorted = dict(sorted(probs.items(), key=operator.itemgetter(1), reverse=True))
    return probs_sorted


@router.post("/model/{model_name}/transcript", status_code=200)
async def transcribe_audio(model_name: str,
                           task: str = Form(..., enum=["transcribe", "translate"]),
                           language_code: str = Form(..., enum=["None"] + sorted(list(whisper.tokenizer.LANGUAGES.keys()))),
                           media_type: str = Form(..., enum=["print", "save"]),
                           format: str = Form(..., enum=["json", "srt", "tsv", "txt", "vtt"]),
                           file: UploadFile = File(...),
                           db = Depends(get_db),
                           request: Request = None):
    """
    Transcribe audio with a Whisper ASR model.
    """
    result = db.search(Query().name == model_name)
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    elif result[0]["downloaded"] == False:
        raise HTTPException(status_code=400, detail="Model not downloaded yet")

    model = whisper.load_model(model_name)

    audio = load_audio(file)
    if language_code == "None":
        options = {
            "task": task,
        }
    else:
        options = {
            "task": task,
            "language": language_code,
        }
    result = model.transcribe(audio, **options)

    # 🔹 Mappa i formati ai writer corretti
    format_map = {
        "json": WriteJSON,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "txt": WriteTXT,
        "vtt": WriteVTT
    }

    if format not in format_map:
        raise HTTPException(status_code=400, detail="Formato non supportato")

    writer_class = format_map[format]
    if writer_class is None:
        raise HTTPException(status_code=500, detail=f"Il formato {format} non è supportato da Whisper")

    # 🔹 Se in modalità print, restituisci il testo nel formato scelto
    if media_type == "print":
        if format == "json":
            return result
        else:
            file_out = io.StringIO()
            writer_class(ResultWriter).write_result(result, file=file_out)
            return file_out.getvalue()  # 🔹 Restituisce il testo formattato come stringa

    # 🔹 Modalità save: salva il file
    filename = file.filename.rsplit(".", 1)[0]
    file_out = io.StringIO()
    writer_class(ResultWriter).write_result(result, file=file_out)
    file_out.seek(0)

    # 🔹 Mappatura corretta dei MIME types
    mime_types = {
        "json": "application/json",
        "srt": "text/srt",
        "tsv": "text/tab-separated-values",
        "txt": "text/plain",
        "vtt": "text/vtt",
    }

    return StreamingResponse(
        file_out,
        media_type=mime_types.get(format, "text/plain"),
        headers={"Content-Disposition": f"attachment; filename={filename}.{format};"}
    )
