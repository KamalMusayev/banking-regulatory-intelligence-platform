"""
Quick verification script for the LLM module.

Run from project root:
    python scripts/verify_llm_module.py
"""

import sys

def main() -> None:
    print("=" * 60)
    print("LLM Module — Verification")
    print("=" * 60)

    # 1. Import all public classes
    print("\n[1] Importing all public classes...")
    try:
        from backend.reguaz.llm import (
            BaseLLMProvider,
            Generator,
            LLMProviderFactory,
            LocalInferenceProvider,
            PromptBuilder,
        )
        print("    ✓ BaseLLMProvider imported")
        print("    ✓ LocalInferenceProvider imported")
        print("    ✓ LLMProviderFactory imported")
        print("    ✓ PromptBuilder imported")
        print("    ✓ Generator imported")
    except Exception as exc:
        print(f"    ✗ Import failed: {exc}")
        sys.exit(1)

    # 2. Verify abstract base class
    print("\n[2] Verifying BaseLLMProvider is abstract...")
    from abc import ABC
    assert issubclass(BaseLLMProvider, ABC), "BaseLLMProvider must be abstract"
    assert hasattr(BaseLLMProvider, "generate"), "Must have generate method"
    assert hasattr(BaseLLMProvider, "model_name"), "Must have model_name property"
    print("    ✓ BaseLLMProvider is a proper ABC with generate() and model_name")

    # 3. Verify LocalInferenceProvider implements BaseLLMProvider
    print("\n[3] Verifying LocalInferenceProvider implements BaseLLMProvider...")
    assert issubclass(LocalInferenceProvider, BaseLLMProvider)
    print("    ✓ LocalInferenceProvider is a BaseLLMProvider subclass")

    # 4. Test PromptBuilder
    print("\n[4] Testing PromptBuilder...")
    pb = PromptBuilder()
    prompt = pb.build_prompt(
        question="Kapital tələbləri nədir?",
        context="Bank kapitalı minimum 50 milyon manat olmalıdır.",
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Kapital tələbləri nədir?" in prompt
    assert "Bank kapitalı minimum 50 milyon manat olmalıdır." in prompt
    print(f"    ✓ Prompt built successfully ({len(prompt)} chars)")
    print(f"    ✓ System prompt length: {len(pb.system_prompt)} chars")

    # 5. Verify Factory
    print("\n[5] Verifying LLMProviderFactory...")
    provider = LLMProviderFactory.get_provider("local")
    assert isinstance(provider, BaseLLMProvider)
    assert isinstance(provider, LocalInferenceProvider)
    assert provider.model_name == "Qwen/Qwen2.5-3B-Instruct"
    print(f"    ✓ Factory returned LocalInferenceProvider (model='{provider.model_name}')")

    # Verify invalid type raises ValueError
    try:
        LLMProviderFactory.get_provider("nonexistent")
        print("    ✗ Should have raised ValueError")
        sys.exit(1)
    except ValueError:
        print("    ✓ Factory raises ValueError for unknown provider types")

    # 6. Verify Generator instantiation
    print("\n[6] Verifying Generator instantiation...")
    gen = Generator(provider_type="local")
    assert gen.model_name == "Qwen/Qwen2.5-3B-Instruct"
    print(f"    ✓ Generator created (model='{gen.model_name}')")

    print("\n" + "=" * 60)
    print("All verifications passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
