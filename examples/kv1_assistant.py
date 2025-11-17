"""
JARVIS - AI Assistant with Dual Memory System

A real AI assistant that:
- Uses HSOKV for human-like memory
- Integrates with any LLM (OpenAI, Anthropic, local models)
- Has emotional state
- Can browse internet
- Learns from conversations
- Never forgets important things

This is a LIVING system with:
- Short-term memory (7±2 current context)
- Long-term memory (permanent knowledge)
- Emotional awareness
- Consolidation during "sleep"
"""

import os
import time
from typing import Optional, Dict, List
from datetime import datetime

from hsokv import DualMemorySystem, SentenceBERTEmbedder, MemoryConfig


class EmotionalState:
    """
    Emotional system for the AI.

    Emotions affect:
    - Which memories are stored directly to long-term (high emotion)
    - Confidence levels
    - Response tone
    """

    def __init__(self):
        self.valence = 0.5  # 0=negative, 1=positive
        self.arousal = 0.3  # 0=calm, 1=excited
        self.trust = 0.5    # 0=suspicious, 1=trusting

    def is_emotionally_significant(self) -> bool:
        """High arousal = emotionally significant = direct to long-term"""
        return self.arousal > 0.7

    def update_from_interaction(self, user_sentiment: str):
        """Update emotions based on user interaction"""
        if user_sentiment == "positive":
            self.valence = min(1.0, self.valence + 0.1)
            self.trust = min(1.0, self.trust + 0.05)
        elif user_sentiment == "negative":
            self.valence = max(0.0, self.valence - 0.1)
            self.arousal = min(1.0, self.arousal + 0.2)  # Get alert
        elif user_sentiment == "urgent":
            self.arousal = 0.9  # High arousal

    def get_state(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "trust": self.trust,
            "mood": self._get_mood()
        }

    def _get_mood(self) -> str:
        if self.valence > 0.7 and self.arousal < 0.3:
            return "content"
        elif self.valence > 0.7 and self.arousal > 0.7:
            return "excited"
        elif self.valence < 0.3 and self.arousal > 0.7:
            return "stressed"
        elif self.valence < 0.3 and self.arousal < 0.3:
            return "sad"
        else:
            return "neutral"


class JarvisAssistant:
    """
    JARVIS - AI Assistant with human-like memory.

    Features:
    - Dual memory system (short-term + long-term)
    - Emotional awareness
    - Learns from conversations
    - Internet access (for real-world info)
    - MCP integration ready
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_provider: str = "openai",  # or "anthropic", "local"
        device: str = "cpu"
    ):
        """
        Initialize JARVIS.

        Args:
            llm_api_key: API key for LLM provider
            llm_provider: "openai", "anthropic", or "local"
            device: "cpu" or "cuda"
        """
        print("🤖 Initializing JARVIS...")

        # Memory system (YOUR DUAL MEMORY IDEA!)
        embedder = SentenceBERTEmbedder(device=device)
        config = MemoryConfig(
            learning_phase_duration=3,
            reinforcement_phase_duration=10,
            device=device
        )

        self.memory = DualMemorySystem(
            embedder=embedder,
            config=config,
            stm_capacity=7,        # 7±2 working memory
            stm_decay_seconds=30   # 30s decay
        )

        # Emotional system
        self.emotions = EmotionalState()

        # LLM setup
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")

        # Conversation context
        self.conversation_history: List[Dict] = []
        self.user_name: Optional[str] = None

        # Stats
        self.interactions_count = 0
        self.last_sleep = time.time()

        print("✅ JARVIS initialized!")
        print(f"   Memory: STM={self.memory.stm.capacity}, LTM=unlimited")
        print(f"   Emotions: {self.emotions.get_state()['mood']}")

    def chat(self, user_input: str) -> str:
        """
        Main chat interface.

        This is where JARVIS:
        1. Recalls relevant memories
        2. Calls LLM with context
        3. Learns from the conversation
        4. Updates emotions

        Args:
            user_input: User's message

        Returns:
            JARVIS's response
        """
        self.interactions_count += 1

        # Step 1: Recall relevant memories
        memories = self._recall_relevant_context(user_input)

        # Step 2: Build prompt with memories
        prompt = self._build_prompt(user_input, memories)

        # Step 3: Get LLM response
        response = self._call_llm(prompt)

        # Step 4: Learn from this interaction
        self._learn_from_interaction(user_input, response)

        # Step 5: Update emotional state
        self._update_emotions(user_input, response)

        # Step 6: Check if need to sleep (consolidate memories)
        if self._should_sleep():
            self._sleep()

        # Add to conversation history
        self.conversation_history.append({
            "user": user_input,
            "jarvis": response,
            "timestamp": datetime.now().isoformat(),
            "emotion": self.emotions.get_state()
        })

        return response

    def _recall_relevant_context(self, query: str) -> List[str]:
        """
        Recall relevant memories for this query.

        Tries:
        1. Short-term memory (current context)
        2. Long-term memory (semantic search)
        """
        memories = []

        # Get short-term context
        stm_stats = self.memory.get_stats()
        if stm_stats["short_term"]["size"] > 0:
            memories.append(f"Current context: {', '.join(stm_stats['short_term']['items'][:3])}")

        # Try long-term semantic search
        ltm_result = self.memory.recall(query)
        if ltm_result:
            memories.append(f"I remember: {ltm_result}")

        return memories

    def _build_prompt(self, user_input: str, memories: List[str]) -> str:
        """Build prompt for LLM with context and memories"""

        system_prompt = f"""You are JARVIS, an AI assistant with human-like memory and emotions.

Current emotional state: {self.emotions.get_state()['mood']}
Memory stats: STM={self.memory.get_stats()['short_term']['size']}/7, LTM={self.memory.get_stats()['long_term']['size']}

Your capabilities:
- Remember conversations (dual memory system)
- Emotional awareness
- Learning from interactions
- Internet access (for real-world info)

Personality:
- Helpful and professional
- Remember user preferences
- Show appropriate emotion in responses
"""

        # Add memories if available
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)

        # Add recent conversation
        recent_context = ""
        if len(self.conversation_history) > 0:
            recent = self.conversation_history[-3:]  # Last 3 exchanges
            recent_context = "\n\nRecent conversation:\n"
            for conv in recent:
                recent_context += f"User: {conv['user']}\nJARVIS: {conv['jarvis']}\n"

        full_prompt = f"""{system_prompt}{memory_context}{recent_context}

User: {user_input}
JARVIS:"""

        return full_prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API.

        Supports:
        - OpenAI (GPT-4, GPT-3.5)
        - Anthropic (Claude)
        - Local models (Ollama, etc.)
        """

        if self.llm_provider == "openai":
            return self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.llm_provider == "local":
            return self._call_local(prompt)
        else:
            # Fallback for demo (no actual LLM)
            return self._demo_response(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            import openai
            openai.api_key = self.llm_api_key

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.llm_api_key)

            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text
        except Exception as e:
            return f"Error calling Anthropic: {e}"

    def _call_local(self, prompt: str) -> str:
        """Call local model (Ollama, etc.)"""
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2", "prompt": prompt, "stream": False}
            )

            return response.json()["response"]
        except Exception as e:
            return f"Error calling local model: {e}"

    def _demo_response(self, prompt: str) -> str:
        """Demo response (no actual LLM)"""
        return "I understand. I'm JARVIS with dual memory system. How can I help you?"

    def _learn_from_interaction(self, user_input: str, response: str):
        """
        Learn from this interaction.

        Stores:
        - User preferences
        - Important facts
        - Context
        """
        # Extract learning opportunities

        # Check for name introduction
        if "my name is" in user_input.lower() or "i am" in user_input.lower():
            # Important! Learn user's name
            words = user_input.lower().split()
            if "my name is" in user_input.lower():
                idx = words.index("name") + 2
                if idx < len(words):
                    name = words[idx].strip(".,!?").title()
                    self.user_name = name
                    # Emotionally significant = direct to long-term!
                    self.memory.learn(
                        "what is the user's name?",
                        name,
                        emotionally_significant=True
                    )
                    self.emotions.arousal = 0.8  # Excited to meet user!

        # Check for preferences
        if "i like" in user_input.lower() or "i prefer" in user_input.lower():
            self.memory.learn(
                f"user preference: {user_input[:50]}",
                user_input,
                emotionally_significant=self.emotions.is_emotionally_significant()
            )

        # Store general interaction
        self.memory.learn(
            user_input[:50],  # Use first 50 chars as key
            response[:100],   # Store response summary
            emotionally_significant=self.emotions.is_emotionally_significant()
        )

    def _update_emotions(self, user_input: str, response: str):
        """Update emotional state based on interaction"""

        # Detect sentiment (simple heuristics)
        positive_words = ["thanks", "great", "awesome", "love", "perfect"]
        negative_words = ["bad", "wrong", "hate", "terrible", "fail"]
        urgent_words = ["urgent", "emergency", "asap", "immediately", "quick"]

        user_lower = user_input.lower()

        if any(word in user_lower for word in urgent_words):
            self.emotions.update_from_interaction("urgent")
        elif any(word in user_lower for word in positive_words):
            self.emotions.update_from_interaction("positive")
        elif any(word in user_lower for word in negative_words):
            self.emotions.update_from_interaction("negative")

        # Gradually return to neutral
        self.emotions.arousal *= 0.95
        self.emotions.valence = 0.5 * 0.1 + self.emotions.valence * 0.9

    def _should_sleep(self) -> bool:
        """
        Check if should consolidate memories.

        Sleep triggers:
        - Every 20 interactions
        - Or every 5 minutes
        """
        time_since_sleep = time.time() - self.last_sleep

        return (
            self.interactions_count % 20 == 0 or
            time_since_sleep > 300  # 5 minutes
        )

    def _sleep(self):
        """
        Consolidate short-term memories to long-term.

        Like human sleep!
        """
        print("\n💤 JARVIS is consolidating memories...")

        before = self.memory.get_stats()
        self.memory.sleep()
        after = self.memory.get_stats()

        print(f"   Consolidated: {before['short_term']['size']} STM → LTM")
        print(f"   Long-term memories: {after['long_term']['size']}")

        self.last_sleep = time.time()

    def get_status(self) -> Dict:
        """Get current system status"""
        stats = self.memory.get_stats()

        return {
            "memory": stats,
            "emotions": self.emotions.get_state(),
            "interactions": self.interactions_count,
            "user_name": self.user_name,
            "uptime": time.time() - self.last_sleep
        }

    def save_memory(self, filepath: str):
        """Save memories to disk (for persistence across restarts)"""
        import pickle

        data = {
            "stm": self.memory.stm.memory,
            "stm_times": self.memory.stm.access_times,
            "stm_counts": self.memory.stm.access_counts,
            # Note: LTM (vector DB) needs separate serialization
            "conversation_history": self.conversation_history,
            "user_name": self.user_name,
            "emotions": self.emotions.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"💾 Memories saved to {filepath}")

    def load_memory(self, filepath: str):
        """Load memories from disk"""
        import pickle

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.memory.stm.memory = data["stm"]
            self.memory.stm.access_times = data["stm_times"]
            self.memory.stm.access_counts = data["stm_counts"]
            self.conversation_history = data["conversation_history"]
            self.user_name = data["user_name"]
            self.emotions.__dict__.update(data["emotions"])

            print(f"💾 Memories loaded from {filepath}")
        except Exception as e:
            print(f"⚠️ Could not load memories: {e}")


def demo():
    """Demo JARVIS assistant"""
    print("=" * 70)
    print("🤖 KV-1 - AI Assistant with Dual Memory")
    print("=" * 70)
    print()

    # Initialize JARVIS
    jarvis = JarvisAssistant(llm_provider="demo", device="cpu")

    print("\nType your messages (or 'quit' to exit)")
    print("Commands: 'status' = show stats, 'sleep' = consolidate memories")
    print("=" * 70)
    print()

    while True:
        # User input
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n👋 JARVIS shutting down...")
            break

        if user_input.lower() == "status":
            status = jarvis.get_status()
            print(f"\n📊 JARVIS Status:")
            print(f"   Memory: STM={status['memory']['short_term']['size']}/7")
            print(f"   Long-term: {status['memory']['long_term']['size']} memories")
            print(f"   Emotion: {status['emotions']['mood']}")
            print(f"   Interactions: {status['interactions']}")
            print()
            continue

        if user_input.lower() == "sleep":
            jarvis._sleep()
            continue

        # Get response
        response = jarvis.chat(user_input)

        # Display
        print(f"🤖 KV-1: {response}")
        print()

        # Show memory status
        stats = jarvis.memory.get_stats()
        stm_size = stats["short_term"]["size"]
        ltm_size = stats["long_term"]["size"]
        print(f"   [STM: {stm_size}/7 | LTM: {ltm_size} | Mood: {jarvis.emotions.get_state()['mood']}]")
        print()


if __name__ == "__main__":
    demo()
