# Data-in-code baseline drain

Issue `6a298a1` removes anonymous ratchet tolerances from the strengthened
destination-based data-in-code gate without discarding the values they protect.

The closing packet records the exact 136-key baseline at integration commit
`470ffe0928712fcdbc7cbaf8f3042b5e919f8008`, including four keys already removed by
integrated sibling work. Every key receives exactly one destination or a documented,
curated exception. Migration fragments and the generated combined manifest live under
`notes/`.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
