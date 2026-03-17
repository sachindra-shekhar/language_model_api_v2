
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes_mcp")
NOTES: dict[str, dict] = {}


@mcp.tool()
async def create_note(title: str, content: str, tags: str = "") -> str:
    """Create a new note with a title, content, and optional comma-separated tags.

    Args:
        title: The title of the note
        content: The main content/body of the note
        tags: Optional comma-separated tags (e.g., "work,urgent,meeting")
    """
    note_id = f"note_{len(NOTES) + 1}"
    NOTES[note_id] = {
        "title": title, "content": content,
        "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        "created_at": datetime.now().isoformat(),
    }
    return json.dumps({"status": "created", "note_id": note_id, "title": title})


@mcp.tool()
async def search_notes(query: str) -> str:
    """Search notes by title, content, or tags.

    Args:
        query: Search term to find in note titles, content, and tags
    """
    query_lower = query.lower()
    matches = []
    for note_id, note in NOTES.items():
        if (query_lower in note["title"].lower() or
            query_lower in note["content"].lower() or
            any(query_lower in tag.lower() for tag in note["tags"])):
            matches.append({"id": note_id, **note})
    if not matches:
        return json.dumps({"message": "No notes found.", "count": 0})
    return json.dumps({"notes": matches, "count": len(matches)}, indent=2, default=str)


@mcp.tool()
async def list_all_notes() -> str:
    """List all saved notes with their titles and IDs."""
    if not NOTES:
        return json.dumps({"message": "No notes yet!", "count": 0})
    summaries = [{"id": nid, "title": n["title"], "tags": n["tags"],
                  "preview": n["content"][:100]} for nid, n in NOTES.items()]
    return json.dumps({"notes": summaries, "count": len(summaries)}, indent=2, default=str)


@mcp.tool()
async def delete_note(note_id: str) -> str:
    """Delete a note by its ID.

    Args:
        note_id: The ID of the note to delete (e.g., note_1)
    """
    if note_id not in NOTES:
        return json.dumps({"error": f"Note {note_id} not found."})
    title = NOTES.pop(note_id)["title"]
    return json.dumps({"status": "deleted", "note_id": note_id, "title": title})


@mcp.resource("notes://all")
def all_notes_resource() -> str:
    """Browse all notes currently stored in the system."""
    return json.dumps(NOTES, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
