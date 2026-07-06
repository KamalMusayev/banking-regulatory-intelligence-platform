import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reguaz.llm import Generator, LLMProviderFactory
provider = LLMProviderFactory.get_provider()

generator = Generator(provider)

context = """
Azərbaycan Respublikasında bankın minimum nizamnamə kapitalı
50 milyon manat olmalıdır.
"""

question = "Minimum kapital neçə manatdır?"

answer = generator.generate_answer(
    question=question,
    context=context,
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)