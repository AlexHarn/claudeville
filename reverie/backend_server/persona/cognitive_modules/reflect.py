"""
Original Author: Joon Sung Park (joonspk@stanford.edu)
Heavily modified for Claudeville (Claude CLI port)

File: reflect.py
Description: This defines the "Reflect" module for generative agents.
"""

import datetime
import sys

sys.path.append("../../")

from persona.cognitive_modules.retrieve import new_retrieve
from persona.prompt_template.run_prompt import (
    run_gpt_prompt_chat_poignancy,
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_event_triple,
    run_gpt_prompt_focal_pt,
    run_gpt_prompt_insight_and_guidance,
    run_gpt_prompt_memo_on_convo,
    run_gpt_prompt_planning_thought_on_convo,
)


def generate_focal_points(persona, n=3):
    nodes = [
        [i.last_accessed, i]
        for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
        if "idle" not in i.embedding_key
    ]

    nodes = sorted(nodes, key=lambda x: x[0])
    nodes = [i for created, i in nodes]

    statements = ""
    for node in nodes[-1 * persona.scratch.importance_ele_n :]:
        statements += node.embedding_key + "\n"

    return run_gpt_prompt_focal_pt(persona, statements, n)[0]


def generate_insights_and_evidence(persona, nodes, n=5):
    statements = ""
    for count, node in enumerate(nodes):
        statements += f"{str(count)}. {node.embedding_key}\n"

    ret = run_gpt_prompt_insight_and_guidance(persona, statements, n)[0]

    try:
        for thought, evi_raw in ret.items():
            evidence_node_id = [nodes[i].node_id for i in evi_raw]
            ret[thought] = evidence_node_id
        return ret
    except Exception:
        return {"this is blank": "node_1"}


def generate_action_event_triple(act_desp, persona):
    """TODO

    INPUT:
      act_desp: the description of the action (e.g., "sleeping")
      persona: The Persona class instance
    OUTPUT:
      a string of emoji that translates action description.
    EXAMPLE OUTPUT:
      "🧈🍞"
    """
    return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_poig_score(persona, event_type, description):
    if "is idle" in description:
        return 1

    if event_type == "event" or event_type == "thought":
        return run_gpt_prompt_event_poignancy(persona, description)[0]
    elif event_type == "chat":
        return run_gpt_prompt_chat_poignancy(persona, persona.scratch.act_description)[
            0
        ]


def generate_planning_thought_on_convo(persona, all_utt):
    return run_gpt_prompt_planning_thought_on_convo(persona, all_utt)[0]


def generate_memo_on_convo(persona, all_utt):
    return run_gpt_prompt_memo_on_convo(persona, all_utt)[0]


def run_reflect(persona):
    """
    Run the actual reflection. We generate the focal points, retrieve any
    relevant nodes, and generate thoughts and insights.

    INPUT:
      persona: Current Persona object
    Output:
      None
    """
    # Reflection requires certain focal points. Generate that first.
    focal_points = generate_focal_points(persona, 3)
    # Retrieve the relevant Nodes object for each of the focal points.
    # <retrieved> has keys of focal points, and values of the associated Nodes.
    retrieved = new_retrieve(persona, focal_points)

    # For each of the focal points, generate thoughts and save it in the
    # agent's memory.
    for focal_pt, nodes in retrieved.items():
        thoughts = generate_insights_and_evidence(persona, nodes, 5)
        for thought, evidence in thoughts.items():
            created = persona.scratch.curr_time
            expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(thought, persona)
            keywords = set([s, p, o])
            thought_poignancy = generate_poig_score(persona, "thought", thought)
            # NOTE: Embeddings removed - just pass thought as the embedding_key identifier
            persona.a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                thought,
                keywords,
                thought_poignancy,
                thought,
                evidence,
            )


def reflection_trigger(persona):
    """
    Given the current persona, determine whether the persona should run a
    reflection.

    Our current implementation checks for whether the sum of the new importance
    measure has reached the set (hyper-parameter) threshold.

    INPUT:
      persona: Current Persona object
    Output:
      True if we are running a new reflection.
      False otherwise.
    """
    if (
        persona.scratch.importance_trigger_curr <= 0
        and persona.a_mem.seq_event + persona.a_mem.seq_thought != []
    ):
        return True
    return False


def reset_reflection_counter(persona):
    """
    We reset the counters used for the reflection trigger.

    INPUT:
      persona: Current Persona object
    Output:
      None
    """
    persona_imt_max = persona.scratch.importance_trigger_max
    persona.scratch.importance_trigger_curr = persona_imt_max
    persona.scratch.importance_ele_n = 0


def reflect(persona):
    """
    The main reflection module for the persona. We first check if the trigger
    conditions are met, and if so, run the reflection and reset any of the
    relevant counters.

    INPUT:
      persona: Current Persona object
    Output:
      None
    """
    if reflection_trigger(persona):
        run_reflect(persona)
        reset_reflection_counter(persona)

    if persona.scratch.chatting_end_time:
        if (
            persona.scratch.curr_time + datetime.timedelta(0, 10)
            == persona.scratch.chatting_end_time
        ):
            all_utt = ""
            if persona.scratch.chat:
                for row in persona.scratch.chat:
                    all_utt += f"{row[0]}: {row[1]}\n"

            evidence = [
                persona.a_mem.get_last_chat(persona.scratch.chatting_with).node_id
            ]

            planning_thought = generate_planning_thought_on_convo(persona, all_utt)
            planning_thought = (
                f"For {persona.scratch.name}'s planning: {planning_thought}"
            )

            created = persona.scratch.curr_time
            expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(planning_thought, persona)
            keywords = set([s, p, o])
            thought_poignancy = generate_poig_score(
                persona, "thought", planning_thought
            )
            # NOTE: Embeddings removed - just pass thought as the embedding_key identifier
            persona.a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                planning_thought,
                keywords,
                thought_poignancy,
                planning_thought,
                evidence,
            )

            memo_thought = generate_memo_on_convo(persona, all_utt)
            memo_thought = f"{persona.scratch.name} {memo_thought}"

            created = persona.scratch.curr_time
            expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
            s, p, o = generate_action_event_triple(memo_thought, persona)
            keywords = set([s, p, o])
            thought_poignancy = generate_poig_score(persona, "thought", memo_thought)
            # NOTE: Embeddings removed - just pass thought as the embedding_key identifier
            persona.a_mem.add_thought(
                created,
                expiration,
                s,
                p,
                o,
                memo_thought,
                keywords,
                thought_poignancy,
                memo_thought,
                evidence,
            )
