FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Install Node.js (version 18.x) and npm
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Install Prettier globally using npm
RUN npm install -g prettier

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory inside the container
WORKDIR /app

RUN mkdir -p /data

# Copy main.py from the host's app directory to the container's /test_app directory
COPY ./app/main.py /app
COPY ./app/function_tasks.py /app


# Command to run the FastAPI app using uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uv", "run", "main.py"]
