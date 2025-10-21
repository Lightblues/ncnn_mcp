# ncnn-mcp

NCNN Whisper Audio Transcription MCP Server

## Prepare model

- download whisper model as [iwiki](https://iwiki.woa.com/p/4016272019);
- prepare the input audio file, e.g. from [openai](https://platform.openai.com/docs/guides/audio?example=audio-in).

reference data structure:
```sh
.
├── data
│   └── alloy.wav
├── model
│   ├── data
│   └── whisper
│       ├── whisper
│       ├── whisper_tiny_decoder.ncnn.bin
│       ├── ...
│       └── whisper_vocab.txt
```

## Run

1. Option 1: insall as a mcp server, ref config file [mcp_config.json](mcp_config.json)
    1. You can test MCP tools with the official tool [Inspector](https://modelcontextprotocol.io/docs/tools/inspector).
2. Option 2: test locally.

```sh
# install dependencies
uv sync
# test whisper binary
uv run python test/run_whisper.py
# test mcp server
uv run python test/test_server.py
```
