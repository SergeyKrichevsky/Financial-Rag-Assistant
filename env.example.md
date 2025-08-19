# .env.example
Template file — copy this to `.env` and fill in your values.

---

## Required Variables

OPENAI_API_KEY=your_openai_api_key_here  
# Your OpenAI API key. Required for Assistant API mode.

---

## Optional Variables

ASSISTANT_ID=your_assistant_id_here  
# Assistant ID (non-secret).  
# If not provided, the app will try to load it from `configs/assistant.meta.json`.

APP_ENV=DEV  
# Options: DEV / PROD.  
# Controls whether the app runs with developer-friendly settings or production mode.

---

## How to Use

1. Duplicate this file and rename it to `.env`.
2. Open the `.env` file and fill in your values.
3. Restart the app to apply the new settings.

---

## Notes

- If you don't want to use `.env`, you can still paste your **OpenAI API key** and **Assistant ID** manually in the app UI.
- `.env` will be **ignored by Git** if `.gitignore` is set up properly.
