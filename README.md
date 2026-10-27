# Mini LLM

## Run with Docker Compose

Copy `.env.example` to `.env`, add your Groq API key, and start the app:

```powershell
docker compose up --build
```

The Compose stack runs the FastAPI app, Redis, and PostgreSQL. PostgreSQL's
`mini_llm_dev` credentials are intended for local development; change them in
`.env` before using the stack in a shared or deployed environment.

## Conversations

Send a question to `POST /ask`:

```json
{
  "question": "What is LangGraph?"
}
```

The response includes a `thread_id`. Send that same ID with later questions to
continue the conversation:

```json
{
  "question": "How does it preserve state?",
  "thread_id": "the-thread-id-from-the-first-response"
}
```

LangGraph's PostgreSQL checkpointer stores each thread's message state. Use
`GET /conversations` to list conversation IDs and `GET
/conversations/{thread_id}` to retrieve a conversation and its messages.
