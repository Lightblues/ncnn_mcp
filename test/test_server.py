#!/usr/bin/env python3
"""Test the MCP server using mcp.client"""

import asyncio
import sys
import pathlib
import traceback
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


async def main():
    """Run the test for the transcribe_audio tool."""
    print("Starting server test...")
    
    DIR = pathlib.Path(__file__).parent.parent
    server_params = StdioServerParameters(
        command="python3",
        args=[str(DIR / "ncnn_mcp" / "server.py")],
        # command="uvx",
        # args=["--from", f"{DIR}", "ncnn-mcp"],
        cwd=DIR
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                print("✓ Connection initialized")

                tools = await session.list_tools()
                assert len(tools.tools) > 0, "No tools found"
                print(f"✓ Found {len(tools.tools)} tools")

                print("\nCalling 'transcribe_audio' tool...")
                result = await session.call_tool(
                    "transcribe_audio",
                    arguments={"file_path": f"{DIR}/data/alloy.wav"}
                )
                
                assert len(result.content) > 0, "No content in result"
                transcribed_text = result.content[0].text
                print(f"Transcribed Text: {transcribed_text}")
                assert isinstance(transcribed_text, str) and len(transcribed_text) > 0, "Transcription is empty"
                assert "[Errno 2] No such file or directory" not in transcribed_text, f"Transcription failed with an error: {transcribed_text}"
                
                print(f"✓ Transcription successful: \"{transcribed_text}\"")

    except Exception as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Test passed!")


if __name__ == "__main__":
    asyncio.run(main())
