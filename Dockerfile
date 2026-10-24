FROM python:3.12-slim-trixie
#COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /app

COPY ./app /app

#RUN uv sync --locked

#RUN pip install uvicorn
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]