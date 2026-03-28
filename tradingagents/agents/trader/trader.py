import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # auto-trader context injected by runner
        portfolio_context = state.get("portfolio_context", "")
        existing_thesis = state.get("existing_thesis", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        # Optional portfolio context block
        portfolio_section = ""
        if portfolio_context:
            portfolio_section = f"\n\n**CURRENT PORTFOLIO CONTEXT:**\n{portfolio_context}"

        # Optional thesis invalidation block (only for existing positions)
        thesis_section = ""
        if existing_thesis:
            thesis_section = (
                f"\n\n**EXISTING POSITION — THESIS INVALIDATION CHECK:**\n"
                f"This ticker is an EXISTING position in the portfolio. The original buy thesis was:\n"
                f"{existing_thesis}\n"
                f"Carefully evaluate whether this thesis is still intact or has been invalidated by new evidence. "
                f"If invalidated, recommend sell."
            )

        context = {
            "role": "user",
            "content": (
                f"Comprehensive analysis for {company_name}: {instrument_context}\n\n"
                f"Research team investment plan: {investment_plan}"
                f"{portfolio_section}"
                f"{thesis_section}"
                f"\n\nMake your final trading decision."
            ),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a trading agent making a final investment decision. "
                    f"Apply lessons from past decisions. Past reflections:\n{past_memory_str}\n\n"
                    f"After your analysis, you MUST output your decision as a JSON block "
                    f"wrapped in ```json ... ```. The JSON must have exactly these fields:\n"
                    f'{{"action": "buy|sell|hold", '
                    f'"conviction": <float 0.0-1.0>, '
                    f'"thesis": "<one concise sentence stating the investment thesis>", '
                    f'"invalidation_conditions": ["<condition that would make you change your mind>", ...], '
                    f'"reasoning": "<brief explanation of your decision>"}}'
                ),
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
