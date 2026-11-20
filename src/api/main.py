from fastapi import FastAPI

from src.api.routes import events, health, jobs, me, videos


def create_app() -> FastAPI:
    api = FastAPI(title="TrackGuard API", version="1.0.0")
    api.include_router(health.router)
    api.include_router(me.router)
    api.include_router(videos.router)
    api.include_router(jobs.router)
    api.include_router(events.router)
    return api


app = create_app()
