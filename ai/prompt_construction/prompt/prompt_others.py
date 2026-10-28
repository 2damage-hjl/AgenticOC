# DEPRECATED: This module is kept for reference only.
# Use prompt_construction.prompt.prompt_builder.build_prompt() instead.
# NPC persona fields are now loaded via prompt_construction.npc.npc_manager.load_npc_config().

from prompt_construction.prompt.template_renderer import render


def get_prompt(
    npc_id,
    state,
    disc,
    instr,
    gift,
    dialogua_instance,
    formatted_history,
    today_conversations,
    ctx,
    player_input,
):
    context = dict(
        npc_name=npc_id,
        attitude=state.get("attitude", "中立"),
        weather=state.get("weather", "晴朗"),
        game_time=state.get("game_time", "未知"),
        location=state.get("location", "未知"),
        player_info=state.get("player_info", "healthy"),
        today_actions=state.get("today_actions", []),
        formatted_history=formatted_history,
        today_conversations=today_conversations,
        persona_text=ctx.get("persona_text", ""),
        summary_text=ctx.get("summary_text", ""),
        event_text=ctx.get("event_text", ""),
        relationship_desc=disc,
        interaction_style=instr,
        gift_rules=gift,
        dialogue_examples=dialogua_instance,
        player_input=player_input,
    )

    sections = [
        "npc/system.jinja2",
        "npc/persona.jinja2",
        "npc/memory.jinja2",
        "npc/context.jinja2",
        "npc/dialogue.jinja2",
    ]

    return "\n\n".join(render(section, **context) for section in sections)
