"""
JARVIS with MCP Integration

This shows how to integrate HSOKV with:
- MCP (Model Context Protocol) for tool use
- Internet access
- File system operations
- Real-time information

For use in custom OS environments.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from hsokv import DualMemorySystem, SentenceBERTEmbedder, MemoryConfig


class MCPToolRegistry:
    """
    Registry for MCP tools that JARVIS can use.

    Tools are functions JARVIS can call to interact with the world:
    - Internet search
    - File operations
    - System commands
    - APIs
    """

    def __init__(self):
        self.tools: Dict[str, callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""

        @self.register("internet_search")
        async def search_internet(query: str) -> str:
            """Search the internet for information"""
            # Integrate with your search API
            return f"Search results for: {query}\n[Results would appear here]"

        @self.register("get_current_time")
        async def get_current_time() -> str:
            """Get current time"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @self.register("read_file")
        async def read_file(filepath: str) -> str:
            """Read a file from system"""
            try:
                with open(filepath, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        @self.register("write_file")
        async def write_file(filepath: str, content: str) -> str:
            """Write to a file"""
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                return f"File written: {filepath}"
            except Exception as e:
                return f"Error writing file: {e}"

        @self.register("get_weather")
        async def get_weather(location: str) -> str:
            """Get weather information"""
            # Integrate with weather API
            return f"Weather in {location}: [API data here]"

        @self.register("set_reminder")
        async def set_reminder(time: str, message: str) -> str:
            """Set a reminder"""
            # Integrate with reminder system
            return f"Reminder set for {time}: {message}"

    def register(self, name: str):
        """Decorator to register a tool"""
        def decorator(func):
            self.tools[name] = func
            return func
        return decorator

    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a registered tool"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        try:
            result = await self.tools[tool_name](**kwargs)
            return result
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())


class JarvisWithMCP:
    """
    JARVIS with full MCP integration.

    Features:
    - Dual memory system
    - MCP tool use
    - Internet access
    - Emotional awareness
    - Persistent memory
    """

    def __init__(self, device: str = "cpu"):
        print("🚀 Initializing JARVIS with MCP...")

        # Memory system
        embedder = SentenceBERTEmbedder(device=device)
        config = MemoryConfig(
            learning_phase_duration=3,
            reinforcement_phase_duration=10,
            device=device
        )

        self.memory = DualMemorySystem(
            embedder=embedder,
            config=config,
            stm_capacity=7,
            stm_decay_seconds=30
        )

        # MCP tools
        self.mcp = MCPToolRegistry()

        # State
        self.conversation_history: List[Dict] = []
        self.user_context: Dict[str, Any] = {}

        print("✅ JARVIS with MCP ready!")
        print(f"   Available tools: {', '.join(self.mcp.list_tools())}")

    async def chat(self, user_input: str) -> str:
        """
        Chat with JARVIS (async version with tool use).

        Flow:
        1. Recall relevant memories
        2. Determine if tools needed
        3. Execute tools if needed
        4. Generate response
        5. Learn from interaction
        """

        # 1. Recall context
        memories = await self._recall_context(user_input)

        # 2. Check if tools needed
        tools_to_use = self._determine_tools(user_input)

        # 3. Execute tools
        tool_results = {}
        if tools_to_use:
            print(f"🔧 Using tools: {', '.join(tools_to_use)}")
            tool_results = await self._execute_tools(tools_to_use, user_input)

        # 4. Generate response
        response = await self._generate_response(
            user_input,
            memories,
            tool_results
        )

        # 5. Learn
        await self._learn(user_input, response, tool_results)

        return response

    async def _recall_context(self, query: str) -> List[str]:
        """Recall relevant memories"""
        memories = []

        # Short-term context
        stats = self.memory.get_stats()
        if stats["short_term"]["size"] > 0:
            memories.append(f"Current: {', '.join(stats['short_term']['items'][:3])}")

        # Long-term recall
        ltm_result = self.memory.recall(query)
        if ltm_result:
            memories.append(f"Remembered: {ltm_result}")

        return memories

    def _determine_tools(self, user_input: str) -> List[str]:
        """
        Determine which tools to use based on user input.

        In production, LLM would decide this.
        Here we use simple heuristics.
        """
        tools = []
        lower = user_input.lower()

        # Simple keyword matching (replace with LLM in production)
        if "search" in lower or "find" in lower or "look up" in lower:
            tools.append("internet_search")

        if "time" in lower or "date" in lower:
            tools.append("get_current_time")

        if "weather" in lower:
            tools.append("get_weather")

        if "remind" in lower or "reminder" in lower:
            tools.append("set_reminder")

        if "read file" in lower or "open file" in lower:
            tools.append("read_file")

        if "write file" in lower or "save file" in lower:
            tools.append("write_file")

        return tools

    async def _execute_tools(self, tools: List[str], user_input: str) -> Dict[str, str]:
        """Execute MCP tools"""
        results = {}

        for tool in tools:
            if tool == "internet_search":
                # Extract search query from input
                query = user_input  # Simplification
                result = await self.mcp.call_tool("internet_search", query=query)
                results[tool] = result

            elif tool == "get_current_time":
                result = await self.mcp.call_tool("get_current_time")
                results[tool] = result

            elif tool == "get_weather":
                # Extract location (simplified)
                result = await self.mcp.call_tool("get_weather", location="current")
                results[tool] = result

            # Add other tools...

        return results

    async def _generate_response(
        self,
        user_input: str,
        memories: List[str],
        tool_results: Dict[str, str]
    ) -> str:
        """
        Generate response using memories and tool results.

        In production, this calls your LLM API.
        """

        # Build context
        context = f"User: {user_input}\n"

        if memories:
            context += f"\nMemories: {'; '.join(memories)}\n"

        if tool_results:
            context += "\nTool results:\n"
            for tool, result in tool_results.items():
                context += f"- {tool}: {result}\n"

        # In production: Call LLM with this context
        # For demo:
        response = f"I understand. "

        if tool_results:
            response += f"I used {len(tool_results)} tool(s) to help. "

        if memories:
            response += "I recall relevant context. "

        response += "How else can I assist?"

        return response

    async def _learn(
        self,
        user_input: str,
        response: str,
        tool_results: Dict[str, str]
    ):
        """Learn from this interaction"""

        # Store interaction in memory
        self.memory.learn(
            user_input[:50],
            response[:100],
            emotionally_significant=bool(tool_results)  # Tool use = important
        )

        # If used tools, remember the results
        for tool, result in tool_results.items():
            self.memory.learn(
                f"{tool}: {user_input[:30]}",
                result[:100],
                emotionally_significant=True
            )

    def consolidate_memories(self):
        """Consolidate memories (call periodically)"""
        self.memory.sleep()


# Integration example for your custom OS
class OSIntegration:
    """
    Example of how to integrate JARVIS into your custom phone OS.

    This shows the architecture for a living system.
    """

    def __init__(self):
        self.jarvis = JarvisWithMCP(device="cpu")  # or "cuda" for phone GPU
        self.is_running = True

    async def start_background_services(self):
        """Start background services"""
        tasks = [
            self._memory_consolidation_service(),
            self._context_awareness_service(),
            self._emotion_regulation_service()
        ]

        await asyncio.gather(*tasks)

    async def _memory_consolidation_service(self):
        """Background service to consolidate memories"""
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            print("💤 Consolidating memories...")
            self.jarvis.consolidate_memories()

    async def _context_awareness_service(self):
        """Monitor context and update JARVIS state"""
        while self.is_running:
            await asyncio.sleep(60)  # Every minute

            # Get phone context
            context = await self._get_phone_context()

            # Update JARVIS context
            self.jarvis.user_context.update(context)

    async def _context_awareness_service(self):
        """Maintain awareness of phone state"""
        while self.is_running:
            await asyncio.sleep(60)

            # Example: Track battery, time, location, etc.
            context = {
                "battery": "85%",
                "time": datetime.now().strftime("%H:%M"),
                "location": "home",  # From GPS
            }

            self.jarvis.user_context.update(context)

    async def _emotion_regulation_service(self):
        """Regulate emotional state over time"""
        while self.is_running:
            await asyncio.sleep(120)  # Every 2 minutes
            # Emotions gradually return to baseline
            # (Implemented in your EmotionalState class)

    async def _get_phone_context(self) -> Dict[str, Any]:
        """Get current phone context"""
        return {
            "battery_level": 85,  # From OS
            "time": datetime.now().strftime("%H:%M"),
            "location": "home",   # From GPS
            "network": "wifi",    # From network manager
            "apps_running": [],   # From app manager
        }

    async def handle_user_message(self, message: str) -> str:
        """Handle user message (main interface)"""
        return await self.jarvis.chat(message)


async def demo():
    """Demo JARVIS with MCP"""
    print("=" * 70)
    print("🚀 JARVIS with MCP Integration")
    print("=" * 70)
    print()

    jarvis = JarvisWithMCP(device="cpu")

    print("\nAvailable MCP tools:")
    for tool in jarvis.mcp.list_tools():
        print(f"  - {tool}")
    print()

    # Test interactions
    test_inputs = [
        "What time is it?",
        "Search for Python tutorials",
        "What's the weather like?",
        "Set a reminder to call mom at 3pm",
    ]

    for user_input in test_inputs:
        print(f"User: {user_input}")
        response = await jarvis.chat(user_input)
        print(f"🤖 KV-1: {response}")
        print()

        # Show memory stats
        stats = jarvis.memory.get_stats()
        print(f"   [STM: {stats['short_term']['size']}/7 | LTM: {stats['long_term']['size']}]")
        print()


if __name__ == "__main__":
    asyncio.run(demo())
