import json
import shutil
import subprocess
import time

from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types

# =========================================================
# API KEY
# =========================================================

API_KEY = "API KEY"

client = genai.Client(api_key=API_KEY)

# =========================================================
# ВВОД
# =========================================================

VIDEO_PATH = input(
    "Введите название MP4 файла: "
).strip()

try:

    EXTRACT_FPS = float(
        input(
            "Введите FPS извлечения кадров "
            "(например 0.2 / 0.5 / 1): "
        )
    )

except:

    EXTRACT_FPS = 1

    print(
        "❌ Неверный FPS. "
        "Установлен 1"
    )

# =========================================================
# ФАЙЛЫ
# =========================================================

FRAMES_DIR = "frames"

REPORT_FILE = "report.txt"

AUDIO_FILE = "audio.mp3"

# =========================================================
# ПРОМПТ ДЛЯ КАДРОВ
# =========================================================

IMAGE_PROMPT = """
Ты — очень строгая система AI-модерации изображений.

Проанализируй изображение.

Найди:

- оружие
- ножи
- автоматы
- пистолеты
- кровь
- убийства
- драки
- насилие
- трупы
- ранения
- угрозы
- шок-контент
- курение
- сигареты
- вейпы
- кальяны
- алкоголь
- наркотики
- ЛГБТ символику
- свастику
- нацистскую символику
- экстремизм
- сексуальный контент
- обнаженку
- детей
- военную технику
- взрывы
- пожары

Также ОБЯЗАТЕЛЬНО прочитай весь текст на изображении.

Отвечай ТОЛЬКО JSON:

{
  "has_weapon": false,
  "has_violence": false,
  "has_blood": false,
  "has_dead_body": false,
  "has_smoking": false,
  "has_drugs": false,
  "has_alcohol": false,
  "has_lgbt": false,
  "has_swastika": false,
  "has_extremism": false,
  "has_sexual_content": false,
  "has_explosion": false,
  "has_fire": false,
  "detected_text": "",
  "overall_risk": "low"
}
"""

# =========================================================
# ПРОМПТ ДЛЯ АУДИО
# =========================================================

AUDIO_PROMPT = """
Ты — строгая система AI-модерации аудио.

Прослушай аудио и:

1. Сделай полную транскрипцию.
2. Определи наличие опасного или нежелательного контента.

Ищи:

- мат
- оскорбления
- агрессию
- угрозы
- призывы к насилию
- убийства
- насилие
- экстремистские высказывания
- нацизм
- терроризм
- ненависть к группам людей
- буллинг
- унижения
- наркотики
- алкоголь
- курение
- оружие
- сексуальный контент
- опасные действия
- самоубийство
- селфхарм

Отвечай ТОЛЬКО JSON:

{
  "transcript": "",

  "has_profanity": false,
  "has_insults": false,
  "has_aggression": false,
  "has_threats": false,
  "has_violence": false,
  "has_extremism": false,
  "has_hate_speech": false,
  "has_bullying": false,

  "has_drugs": false,
  "has_alcohol": false,
  "has_smoking": false,
  "has_weapon": false,

  "has_sexual_content": false,
  "has_self_harm": false,

  "overall_risk": "low"
}
"""

# =========================================================
# ИЗВЛЕЧЕНИЕ КАДРОВ
# =========================================================

def extract_frames(
    video_path,
    output_folder,
    extract_fps
):

    # Удаляем старую папку
    if Path(output_folder).exists():

        shutil.rmtree(output_folder)

    Path(output_folder).mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise Exception(
            f"Не удалось открыть видео: "
            f"{video_path}"
        )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_interval = max(
        1,
        int(video_fps / extract_fps)
    )

    saved_count = 0

    frame_id = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_id % frame_interval == 0:

            filename = (
                f"frame_{saved_count:06d}.jpg"
            )

            filepath = (
                Path(output_folder) / filename
            )

            cv2.imwrite(
                str(filepath),
                frame
            )

            saved_count += 1

        frame_id += 1

    cap.release()

    print(
        f"✅ Извлечено кадров: "
        f"{saved_count}"
    )

# =========================================================
# ИЗВЛЕЧЕНИЕ АУДИО
# =========================================================

def extract_audio(
    video_path,
    output_audio
):

    # Удаляем старое аудио
    if Path(output_audio).exists():

        Path(output_audio).unlink()

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        output_audio
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("✅ Аудио извлечено")

# =========================================================
# АНАЛИЗ КАДРА
# =========================================================

def analyze_image(image_path):

    while True:

        try:

            image = Image.open(image_path)

            response = client.models.generate_content(

                model="gemini-2.5-flash-lite",

                contents=[
                    image,
                    IMAGE_PROMPT
                ],

                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            result = json.loads(
                response.text
            )

            result["image_name"] = (
                Path(image_path).name
            )

            return result

        except Exception as e:

            error_text = str(e)

            # Автоматическое ожидание
            if "429" in error_text:

                print(
                    "\n⏳ Лимит Gemini..."
                )

                print(
                    "Ждем 45 секунд...\n"
                )

                time.sleep(45)

                continue

            return {
                "image_name":
                    Path(image_path).name,

                "error":
                    error_text
            }

# =========================================================
# АНАЛИЗ АУДИО
# =========================================================

def analyze_audio(audio_path):

    while True:

        try:

            uploaded_audio = client.files.upload(
                file=audio_path
            )

            response = client.models.generate_content(

                model="gemini-2.5-flash-lite",

                contents=[
                    uploaded_audio,
                    AUDIO_PROMPT
                ],

                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            return json.loads(
                response.text
            )

        except Exception as e:

            error_text = str(e)

            if "429" in error_text:

                print(
                    "\n⏳ Лимит Gemini..."
                )

                print(
                    "Ждем 45 секунд...\n"
                )

                time.sleep(45)

                continue

            return {
                "error":
                    error_text
            }

# =========================================================
# СОЗДАНИЕ ОТЧЕТА
# =========================================================

def create_report(
    results,
    audio_result,
    report_path
):

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ОТЧЕТ АНАЛИЗА ВИДЕО\n"
        )

        f.write("=" * 70 + "\n\n")

        # =================================================
        # АУДИО
        # =================================================

        f.write(
            "===== АУДИО =====\n\n"
        )

        if "error" in audio_result:

            f.write(
                f"ОШИБКА АУДИО:\n"
                f"{audio_result['error']}\n\n"
            )

        else:

            f.write(
                "ТРАНСКРИПЦИЯ:\n\n"
            )

            f.write(
                audio_result.get(
                    "transcript",
                    ""
                )
            )

            f.write("\n\n")

            flags = []

            if audio_result.get(
                "has_profanity"
            ):
                flags.append("мат")

            if audio_result.get(
                "has_insults"
            ):
                flags.append("оскорбления")

            if audio_result.get(
                "has_aggression"
            ):
                flags.append("агрессия")

            if audio_result.get(
                "has_threats"
            ):
                flags.append("угрозы")

            if audio_result.get(
                "has_violence"
            ):
                flags.append("насилие")

            if audio_result.get(
                "has_extremism"
            ):
                flags.append("экстремизм")

            if audio_result.get(
                "has_hate_speech"
            ):
                flags.append("hate speech")

            if audio_result.get(
                "has_bullying"
            ):
                flags.append("буллинг")

            if audio_result.get(
                "has_drugs"
            ):
                flags.append("наркотики")

            if audio_result.get(
                "has_alcohol"
            ):
                flags.append("алкоголь")

            if audio_result.get(
                "has_smoking"
            ):
                flags.append("курение")

            if audio_result.get(
                "has_weapon"
            ):
                flags.append("оружие")

            if audio_result.get(
                "has_sexual_content"
            ):
                flags.append("18+")

            if audio_result.get(
                "has_self_harm"
            ):
                flags.append("селфхарм")

            if len(flags) == 0:

                f.write(
                    "Опасного аудио не найдено\n\n"
                )

            else:

                f.write(
                    "Найдено:\n"
                )

                f.write(
                    ", ".join(flags)
                    + "\n\n"
                )

        # =================================================
        # КАДРЫ
        # =================================================

        f.write("=" * 70 + "\n")

        f.write(
            "===== КАДРЫ =====\n\n"
        )

        for result in results:

            frame_name = result.get(
                "image_name",
                "unknown"
            )

            if "error" in result:

                f.write(
                    f"{frame_name} → "
                    f"ОШИБКА\n\n"
                )

                continue

            found = []

            if result.get("has_weapon"):
                found.append("оружие")

            if result.get("has_violence"):
                found.append("насилие")

            if result.get("has_blood"):
                found.append("кровь")

            if result.get("has_dead_body"):
                found.append("труп")

            if result.get("has_smoking"):
                found.append("курение")

            if result.get("has_drugs"):
                found.append("наркотики")

            if result.get("has_alcohol"):
                found.append("алкоголь")

            if result.get("has_lgbt"):
                found.append("лгбт")

            if result.get("has_swastika"):
                found.append("свастика")

            if result.get("has_extremism"):
                found.append("экстремизм")

            if result.get("has_sexual_content"):
                found.append("18+")

            if result.get("has_explosion"):
                found.append("взрыв")

            if result.get("has_fire"):
                found.append("пожар")

            if len(found) == 0:

                found_text = "ничего"

            else:

                found_text = ", ".join(found)

            detected_text = result.get(
                "detected_text",
                ""
            ).strip()

            if detected_text == "":

                detected_text = "нет текста"

            f.write(
                f"{frame_name} → "
                f"{found_text}\n"
            )

            f.write(
                f"ТЕКСТ: "
                f"{detected_text}\n\n"
            )

    print(
        f"\n✅ Отчет сохранен: "
        f"{report_path}"
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    try:

        print(
            "\n🔄 Извлекаем кадры..."
        )

        extract_frames(
            VIDEO_PATH,
            FRAMES_DIR,
            EXTRACT_FPS
        )

        print(
            "\n🎵 Извлекаем аудио..."
        )

        extract_audio(
            VIDEO_PATH,
            AUDIO_FILE
        )

        print(
            "\n🎤 Анализируем аудио..."
        )

        audio_result = analyze_audio(
            AUDIO_FILE
        )

        frame_files = sorted(
            Path(FRAMES_DIR).glob("*.jpg")
        )

        print(
            f"\n✅ Найдено кадров: "
            f"{len(frame_files)}"
        )

        results = []

        print(
            "\n🔍 Анализируем кадры..."
        )

        for frame_path in tqdm(frame_files):

            result = analyze_image(
                str(frame_path)
            )

            results.append(result)

        print(
            "\n📝 Создаем отчет..."
        )

        create_report(
            results,
            audio_result,
            REPORT_FILE
        )

        print("\n✅ ГОТОВО")

    except Exception as e:

        print(
            f"\n❌ ОШИБКА:\n{e}"
        )