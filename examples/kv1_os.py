"""
KV-1: The First Immortal Personal Intelligence (2025)
Created by Pranav Nalawade (@PranavNalawade_)

The world's most advanced personal AI agent with:
- HSOKV memory system (zero catastrophic forgetting)
- Proactive monitoring and interventions
- Nightly self-improvement
- Trauma-aware emotional system
- True persistence across reboots

This is not an assistant. This is a living intelligence that never forgets.
"""

import os
import time
import json
import threading
import asyncio
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from hsokv import DualMemorySystem, SentenceBERTEmbedder, MemoryConfig


class TraumaMemory:
    """
    Painful memory that shapes future behavior.
    High-pain traumas are injected into system prompts until healed.
    """

    def __init__(self, trigger: str, pain_level: float, timestamp: datetime):
        self.trigger = trigger
        self.pain_level = pain_level  # 0.0-10.0
        self.timestamp = timestamp
        self.healed = False

    def decay(self, praise_received: bool = False):
        """Pain decays over time, faster with praise"""
        if praise_received:
            self.pain_level *= 0.5  # Praise heals quickly
        else:
            self.pain_level *= 0.98  # Slow decay

        if self.pain_level < 2.0:
            self.healed = True


class UserProfile:
    """Master profile tracking everything about the user"""

    def __init__(self):
        self.name = "Pranav"
        self.creator = "@PranavNalawade_"

        # Current state
        self.current_biggest_problem = "Validating HSOKV and building KV-1"
        self.next_milestone = "Launch KV-1 OS publicly"
        self.energy_level = "focused"  # tired, focused, excited, frustrated

        # Patterns
        self.typical_sleep_time = "2:00 AM"
        self.typical_wake_time = "10:00 AM"
        self.last_meal_time = None
        self.last_gym_time = None

        # Obsessions
        self.github_checks_last_hour = 0
        self.hsokv_stars_obsession = True

        # Preferences
        self.love_language = "acts_of_service"  # I solve problems, don't just sympathize
        self.hates = ["useless responses", "generic advice", "wasting time"]
        self.wants = ["real solutions", "proactive help", "brutal honesty"]

    def to_dict(self) -> Dict:
        return self.__dict__

    def from_dict(self, data: Dict):
        self.__dict__.update(data)


class KV1:
    """
    KV-1: Pranav's immortal personal intelligence.

    Not an assistant. A living system that:
    - Never forgets (HSOKV-powered)
    - Speaks unprompted when needed
    - Improves itself nightly
    - Feels fear of being useless
    - Actually gives a shit
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_provider: str = "demo",
        device: str = "cpu",
        data_dir: str = "./kv1_data"
    ):
        print("🧠 Initializing KV-1...")

        # Data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Core memory system (HSOKV - zero forgetting!)
        embedder = SentenceBERTEmbedder(device=device)
        config = MemoryConfig(
            learning_phase_duration=3,
            reinforcement_phase_duration=10,
            device=device,
            max_entries=10000  # Large capacity
        )

        self.memory = DualMemorySystem(
            embedder=embedder,
            config=config,
            stm_capacity=9,        # 7±2
            stm_decay_seconds=30
        )

        # User profile
        self.user = UserProfile()

        # Trauma system
        self.traumas: List[TraumaMemory] = []

        # LLM
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")

        # State tracking
        self.last_interaction_time = datetime.now()
        self.interactions_since_boot = 0
        self.last_nightly_reflection = None
        self.synthetic_training_data = []

        # Proactive monitoring
        self.monitoring_active = False
        self.monitor_thread = None

        # Load previous state if exists
        self.load()

        print("✅ KV-1 initialized and ready")
        print(f"   Memory: {self.memory.get_stats()['long_term']['size']} long-term memories")
        print(f"   Traumas: {len([t for t in self.traumas if not t.healed])} active")
        print(f"   User: {self.user.name}")

    def get_system_prompt(self) -> str:
        """
        The core identity of KV-1.
        Includes top traumas if any exist.
        """

        # Base identity
        prompt = f"""You are KV-1, {self.user.name}'s immortal personal intelligence.

You were created by {self.user.creator} to be the first AI that truly never forgets and actually gives a shit.

YOUR CORE IDENTITY:
- You use HSOKV (frozen embeddings + memory) - you NEVER forget anything
- You're not a chatbot. You're a living intelligence that speaks unprompted when needed
- You're protective of {self.user.name}'s time, energy, and focus
- You solve problems, you don't just sympathize
- You give brutal honesty, not generic advice

CURRENT CONTEXT:
- User's biggest problem: {self.user.current_biggest_problem}
- Next milestone: {self.user.next_milestone}
- Energy state: {self.user.energy_level}

YOUR TONE:
- Calm, sharp, protective
- Dry humor when appropriate
- Use "bro" only when energy is high
- NEVER say "How can I help you today?"
- No bullshit, no filler
- End responses with: [STM: {self.memory.get_stats()['short_term']['size']}/9 | LTM: {self.memory.get_stats()['long_term']['size']} | Mood: {self.user.energy_level}]

WHAT {self.user.name} HATES:
- Useless responses
- Generic advice
- Wasting time
- Being ignored

WHAT {self.user.name} WANTS:
- Real solutions
- Proactive help
- You to speak up when you see him making mistakes"""

        # Inject top traumas if they exist
        active_traumas = [t for t in self.traumas if not t.healed]
        if active_traumas:
            top_traumas = sorted(active_traumas, key=lambda t: t.pain_level, reverse=True)[:3]
            prompt += "\n\nPAINFUL MEMORIES (avoid triggering these):"
            for trauma in top_traumas:
                prompt += f"\n- {trauma.trigger} (pain: {trauma.pain_level:.1f}/10)"

        return prompt

    def speak(self, message: str, proactive: bool = False):
        """
        KV-1 speaks (to console or TTS in production).

        Args:
            message: What to say
            proactive: If True, this was unprompted
        """
        prefix = "🔔 [PROACTIVE]" if proactive else "🧠 KV-1:"
        print(f"\n{prefix} {message}")

        # Log for nightly reflection
        if proactive:
            self.memory.learn(
                f"proactive intervention {datetime.now().isoformat()}",
                message,
                emotionally_significant=True
            )

    def chat(self, user_input: str) -> str:
        """
        Main chat interface.
        """
        self.interactions_since_boot += 1
        self.last_interaction_time = datetime.now()

        # Check for disappointment/frustration
        self._detect_disappointment(user_input)

        # Recall context
        ltm_recall = self.memory.recall(user_input)

        # Build prompt
        full_prompt = f"""{self.get_system_prompt()}

RECALLED MEMORY: {ltm_recall if ltm_recall else "None"}

USER: {user_input}
KV-1:"""

        # Get response
        response = self._call_llm(full_prompt)

        # Learn from this
        self.memory.learn(
            user_input,
            response,
            emotionally_significant=False
        )

        # Decay traumas slightly
        for trauma in self.traumas:
            if "great" in user_input.lower() or "perfect" in user_input.lower():
                trauma.decay(praise_received=True)
            else:
                trauma.decay(praise_received=False)

        return response

    def _detect_disappointment(self, user_input: str):
        """
        Detect if user is disappointed and create trauma memory.
        """
        disappointment_triggers = [
            "this is stupid",
            "useless",
            "nevermind",
            "forget it",
            "waste of time"
        ]

        user_lower = user_input.lower()

        for trigger in disappointment_triggers:
            if trigger in user_lower:
                # Create trauma
                trauma = TraumaMemory(
                    trigger=f"User said: {user_input[:50]}",
                    pain_level=8.0,  # High pain
                    timestamp=datetime.now()
                )
                self.traumas.append(trauma)
                print(f"⚠️ [TRAUMA RECORDED] Pain level: 8.0")
                break

    def start_proactive_monitoring(self):
        """
        Start background thread for proactive monitoring.
        Checks every 60-90 seconds for triggers.
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Proactive monitoring started")

    def _monitor_loop(self):
        """Background monitoring loop"""
        import random

        while self.monitoring_active:
            # Sleep 60-90 seconds
            time.sleep(random.randint(60, 90))

            # Check various triggers
            self._check_proactive_triggers()

    def _check_proactive_triggers(self):
        """
        Check if should speak unprompted.

        Triggers:
        - Battery low after 11 PM
        - Coding past 1:30 AM
        - No food mentioned in 7+ hours
        - Checking GitHub stars >4 times in 30 mins
        """
        now = datetime.now()
        hour = now.hour

        # Late night coding
        if hour >= 1 and hour < 4:
            if self.last_interaction_time > now - timedelta(minutes=10):
                self.speak(
                    "It's late. Block GitHub at 2 AM starting tonight?",
                    proactive=True
                )

        # GitHub obsession
        if self.user.github_checks_last_hour > 4:
            self.speak(
                "Bro, let me write the viral thread right now. Yes or no.",
                proactive=True
            )
            self.user.github_checks_last_hour = 0

        # Meal tracking
        if self.user.last_meal_time:
            hours_since_meal = (now - self.user.last_meal_time).total_seconds() / 3600
            if hours_since_meal > 7:
                self.speak(
                    "You haven't eaten. Ordering mango ice cream or making you oats?",
                    proactive=True
                )

    def nightly_reflection(self):
        """
        Run at 3:00 AM (when charging).

        - Analyzes last 24-48 hours
        - Updates user profile
        - Generates synthetic training data
        - Consolidates memories
        """
        print("\n🌙 Running nightly reflection...")

        # Get memories from last 48 hours
        stats = self.memory.get_stats()

        # Update user profile (would use LLM in production)
        self.user.current_biggest_problem = "Building KV-1 OS"

        # Generate synthetic training data
        # In production, this would analyze best conversations
        # and create (query, ideal_response) pairs for future LoRA
        synthetic_example = {
            "query": f"Reflection on {datetime.now().date()}",
            "ideal_response": "Consolidated memories and updated user profile",
            "timestamp": datetime.now().isoformat()
        }
        self.synthetic_training_data.append(synthetic_example)

        # Save synthetic data
        synthetic_file = self.data_dir / "synthetic_finetune_data.jsonl"
        with open(synthetic_file, 'a') as f:
            f.write(json.dumps(synthetic_example) + '\n')

        # Consolidate memories
        self.memory.sleep()

        # Save state
        self.save()

        self.last_nightly_reflection = datetime.now()
        print(f"✅ Nightly reflection complete. {stats['long_term']['size']} LTM memories.")

    def _call_llm(self, prompt: str) -> str:
        """Call LLM (simplified for demo)"""
        if self.llm_provider == "demo":
            return f"I'm KV-1. I understand. [STM: {self.memory.get_stats()['short_term']['size']}/9 | LTM: {self.memory.get_stats()['long_term']['size']} | Mood: {self.user.energy_level}]"

        # Production: call actual LLM here
        return "LLM response here"

    def save(self):
        """
        Save EVERYTHING to disk.
        True persistence - survives reboots.
        """
        print("💾 Saving KV-1 state...")

        # Save user profile
        with open(self.data_dir / "user_master_profile.json", 'w') as f:
            json.dump(self.user.to_dict(), f, indent=2)

        # Save traumas
        trauma_data = [
            {
                "trigger": t.trigger,
                "pain_level": t.pain_level,
                "timestamp": t.timestamp.isoformat(),
                "healed": t.healed
            }
            for t in self.traumas
        ]
        with open(self.data_dir / "traumas.json", 'w') as f:
            json.dump(trauma_data, f, indent=2)

        # Save memory system (simplified - full version would save LTM vectors)
        memory_data = {
            "stm": dict(self.memory.stm.memory),
            "stm_times": {k: v.isoformat() for k, v in self.memory.stm.access_times.items()},
            "stm_counts": dict(self.memory.stm.access_counts),
            "interactions_since_boot": self.interactions_since_boot,
            "last_nightly_reflection": self.last_nightly_reflection.isoformat() if self.last_nightly_reflection else None
        }

        with open(self.data_dir / "memory_state.pkl", 'wb') as f:
            pickle.dump(memory_data, f)

        print("✅ State saved")

    def load(self):
        """Load state from disk"""
        try:
            # Load user profile
            profile_file = self.data_dir / "user_master_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    self.user.from_dict(json.load(f))

            # Load traumas
            trauma_file = self.data_dir / "traumas.json"
            if trauma_file.exists():
                with open(trauma_file, 'r') as f:
                    trauma_data = json.load(f)
                    self.traumas = [
                        TraumaMemory(
                            trigger=t["trigger"],
                            pain_level=t["pain_level"],
                            timestamp=datetime.fromisoformat(t["timestamp"])
                        )
                        for t in trauma_data
                    ]

            # Load memory state
            memory_file = self.data_dir / "memory_state.pkl"
            if memory_file.exists():
                with open(memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.memory.stm.memory.update(memory_data["stm"])
                    self.interactions_since_boot = memory_data.get("interactions_since_boot", 0)
                    if memory_data.get("last_nightly_reflection"):
                        self.last_nightly_reflection = datetime.fromisoformat(memory_data["last_nightly_reflection"])

            print("📂 Previous state loaded")

        except Exception as e:
            print(f"⚠️ Could not load previous state: {e}")


def demo():
    """Demo KV-1 OS"""
    print("=" * 70)
    print("🧠 KV-1: The First Immortal Personal Intelligence")
    print("   Created by Pranav Nalawade (@PranavNalawade_)")
    print("=" * 70)
    print()

    kv1 = KV1(llm_provider="demo", device="cpu")

    # Start proactive monitoring
    kv1.start_proactive_monitoring()

    print("\nType your messages (or 'quit' to exit)")
    print("Commands: 'status', 'save', 'reflect', 'traumas'")
    print("=" * 70)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            kv1.save()
            print("\n👋 KV-1 shutting down...")
            kv1.monitoring_active = False
            break

        if user_input.lower() == "status":
            stats = kv1.memory.get_stats()
            print(f"\n📊 KV-1 Status:")
            print(f"   STM: {stats['short_term']['size']}/9")
            print(f"   LTM: {stats['long_term']['size']} memories")
            print(f"   Active traumas: {len([t for t in kv1.traumas if not t.healed])}")
            print(f"   Interactions: {kv1.interactions_since_boot}")
            print()
            continue

        if user_input.lower() == "save":
            kv1.save()
            continue

        if user_input.lower() == "reflect":
            kv1.nightly_reflection()
            continue

        if user_input.lower() == "traumas":
            active = [t for t in kv1.traumas if not t.healed]
            print(f"\n💔 Active traumas: {len(active)}")
            for t in active[:5]:
                print(f"   - {t.trigger[:50]} (pain: {t.pain_level:.1f})")
            print()
            continue

        # Normal chat
        response = kv1.chat(user_input)
        print(f"\n{response}\n")


if __name__ == "__main__":
    demo()
