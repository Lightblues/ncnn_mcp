#!/usr/bin/env python3
"""NCNN Whisper Audio Transcription MCP Server"""

import pathlib
import subprocess
import hashlib
import os
from datetime import datetime
import random
import string
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Allow overriding the model directory via environment variable
# This is useful when running via uvx where the package is installed
# in a different location than the source code
MODEL_DIR = os.environ.get("NCNN_MCP_MODEL_DIR")
if MODEL_DIR:
    DIR = pathlib.Path(MODEL_DIR)
else:
    DIR = pathlib.Path(__file__).parent.parent

def transcribe_audio(file_path: str) -> str:
    """Transcribe audio file to text using NCNN Whisper model."""
    try:
        # Get absolute paths - now we're in ncnn_mcp package
        build_examples_dir = DIR / "model"
        data_dir = build_examples_dir / "data"

        # Create data directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename: date+hash+random.wav
        file_hash = hashlib.md5(pathlib.Path(file_path).read_bytes()).hexdigest()[:8]
        date_str = datetime.now().strftime("%Y%m%d")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        output_wav = f"{date_str}_{file_hash}_{random_suffix}.wav"
        output_wav_path = data_dir / output_wav
        
        # Step 1: Convert to PCM format using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-i", file_path, "-vn", "-c:a", "pcm_s16le", 
            "-ac", "1", "-ar", "16000", "-fflags", "bitexact", 
            str(output_wav_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        # Step 2: Run whisper transcription
        whisper_dir = build_examples_dir / "whisper"
        whisper_executable = whisper_dir / "whisper"
        whisper_cmd = [str(whisper_executable), str(output_wav_path)]
        result = subprocess.run(whisper_cmd, check=True, capture_output=True, 
                               text=True, cwd=whisper_dir)
        
        # Parse output
        combined_output = "\n".join(filter(None, [result.stdout, result.stderr]))
        
        if_found, output = False, ""
        for line in combined_output.splitlines():
            line = line.strip()
            if line.startswith("text ="):
                if_found = True
                parts = line.split("=", 1)
                if len(parts) > 1:
                    output += parts[1].strip()
            else:
                if if_found:
                    output += "\n" + line
        if if_found:
            return output.strip()
        else:
            # fallback (or return null?)
            return combined_output.strip()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}"


def main():
    """Entry point for uvx."""
    asyncio.run(run_server())


async def run_server():
    """Run the MCP server."""
    server = Server("ncnn-whisper")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="transcribe_audio",
                description="Transcribe audio file to text using NCNN Whisper model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the audio file to transcribe"
                        }
                    },
                    "required": ["file_path"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "transcribe_audio":
            file_path = arguments["file_path"]
            result = transcribe_audio(file_path)
            return [TextContent(type="text", text=result)]
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
