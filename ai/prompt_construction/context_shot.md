# Project Rules
环境路径：d:/anaconda3/envs/ai_env
- This project builds a Stardew Valley dialogue agent.
- Raw unpacked game data is under data/game_scripts and must never be modified.
- Dialogue and Marriage are the current scope.
- Extract canonical records first; do not build embeddings before canonical extraction is stable.
- One canonical dialogue may have multiple language texts under the same canonical_id.
- Author keys must be preserved as author_key.
- Global id format: Dialogue/{dialogue_type}/{character}:{author_key}.
- Marriage dialogue must get required_flags = relationship.married_to.{character}.
- Key parser handles season, weekday, heart_min, weather, special keys.
- LLM must not decide hard conditions such as marriage, heart_min, route, or required_flags.
- route and required_flags are metadata filters and must not be included in embedding_text.
- LanceDB retrieval returns candidates only, not final few-shot.