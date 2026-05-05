def get_mitigation_strategies(top_features):
    """
    Maps top stress factors to highly specific, research-backed medical interventions.
    Contains 20+ specialized protocols.
    """
    strategy_map = {
        "Have you recently experienced stress in your life?": {
            "title": "Cortisol Regulation Protocol",
            "advice": "Immediate implementation of Box Breathing (4s inhale, 4s hold, 4s exhale, 4s hold) to reset the autonomic nervous system.",
            "icon": "🧘"
        },
        "Have you noticed a rapid heartbeat or palpitations?": {
            "title": "Vagal Nerve Stimulation",
            "advice": "Cold water face immersion (Mammalian Dive Reflex) or humming to stimulate the vagus nerve and reduce heart rate variability issues.",
            "icon": "💓"
        },
        "Have you been dealing with anxiety or tension recently?": {
            "title": "Cognitive Reframing (CBT-S)",
            "advice": "Utilize the 'Worst-Case/Best-Case/Likely-Case' evidence-based worksheet to deconstruct catastrophic thinking patterns.",
            "icon": "🛡️"
        },
        "Do you face any sleep problems or difficulties falling asleep?": {
            "title": "Circadian Synchronization",
            "advice": "Limit blue light exposure 90m before sleep. Use magnesium glycinate or Chamomile tea to support GABA production naturally.",
            "icon": "🌙"
        },
        "Have you been getting headaches more often than usual?": {
            "title": "Neuromuscular Release",
            "advice": "Implement a 20-minute dark-room session. Apply sub-occipital release techniques to alleviate tension-type headaches.",
            "icon": "💆"
        },
        "Do you get irritated easily?": {
            "title": "Amygdala Hijack Prevention",
            "advice": "Practice the '90-Second Rule'—allow the chemical surge of emotion to dissipate before reacting or making decisions.",
            "icon": "⚖️"
        },
        "Do you have trouble concentrating on your academic tasks?": {
            "title": "Dopamine Fasting & Deep Work",
            "advice": "Batch-process notifications. Use 90-minute concentrated 'Deep Work' blocks followed by 15-minute screen-free breaks.",
            "icon": "🎯"
        },
        "Have you been feeling sadness or low mood?": {
            "title": "Behavioral Activation",
            "advice": "Schedule 'Non-Zero Days'. Commit to just 5 minutes of one meaningful activity to bypass the inertia of low mood.",
            "icon": "☀️"
        },
        "Do you often feel lonely or isolated?": {
            "title": "Social Micro-Dosing",
            "advice": "Engage in 'weak tie' interactions (e.g., small talk with a barista). These small social hits are clinically proven to reduce isolation.",
            "icon": "🤝"
        },
        "Do you feel overwhelmed with your academic workload?": {
            "title": "Eisenhower Matrix Deployment",
            "advice": "Categorize tasks by Urgency vs. Importance. Delegate or delete non-essential tasks to lower the cognitive burden.",
            "icon": "📚"
        },
        "Are you in competition with your peers, and does it affect you?": {
            "title": "Intrinsic Value Alignment",
            "advice": "Shift focus to 'Mastery Goals' (learning for self) rather than 'Performance Goals' (ranking against others).",
            "icon": "🏁"
        },
        "Do you find that your relationship often causes you stress?": {
            "title": "Interpersonal Boundary Setting",
            "advice": "Implement 'The 20-Minute Decompress'—intentional space between academic activities and relationship interactions.",
            "icon": "💬"
        },
        "Are you facing any difficulties with your professors or instructors?": {
            "title": "Advocacy & Negotiation",
            "advice": "Prepare a 'Solutions-Based' memo before meetings. Frame concerns as opportunities for academic growth.",
            "icon": "🎓"
        },
        "Is your working environment unpleasant or stressful?": {
            "title": "Ergonomic & Sensory Audit",
            "advice": "Introduce 'Biophilic Elements' (plants) and white noise to mask acoustic stressors in the environment.",
            "icon": "🏢"
        },
        "Do you struggle to find time for relaxation and leisure activities?": {
            "title": "Time Block Radicalization",
            "advice": "Hard-code 'Rest Debt Repayment' into your calendar. Treat leisure as a non-negotiable medical prescription.",
            "icon": "🎡"
        },
        "Is your hostel or home environment causing you difficulties?": {
            "title": "Sanctuary Creation",
            "advice": "Establish clear 'Noise and Space' contracts with co-habitants. Use acoustic panels or white noise machines.",
            "icon": "🏠"
        },
        "Do you lack confidence in your academic performance?": {
            "title": "Competence-Confidence Loop",
            "advice": "Review previous 'wins'. Record all academic successes, no matter how small, to build a resilient evidence base for yourself.",
            "icon": "💎"
        },
        "Academic and extracurricular activities conflicting for you?": {
            "title": "Strategic De-Commitment",
            "advice": "Evaluate the 80/20 rule of your activities. Which 20% of your extracurriculars provide 80% of your fulfillment?",
            "icon": "🤹"
        },
        "Have you gained/lost weight?": {
            "title": "Metabolic Homeostasis",
            "advice": "Prioritize high-protein, low-inflammatory diets. Avoid 'stress-eating' by implementing the 10-minute water-first rule.",
            "icon": "⚖️"
        },
        "Gender": {
            "title": "Hormonal Balance",
            "advice": "Consider gender-specific physiological responses to stress and consult a healthcare provider for endocrine support.",
            "icon": "🧬"
        },
        "Age": {
            "title": "Life Stage Calibration",
            "advice": "Align academic expectations with developmental life stages. Focus on long-term sustainability over short-term burnout.",
            "icon": "⏳"
        }
    }

    recommendations = []
    for feat, impact in top_features:
        if feat in strategy_map:
            # Add a dynamic 'Intensity' label based on SHAP impact
            rec = strategy_map[feat].copy()
            if impact > 0.1: rec['intensity'] = "Critical Impact"
            elif impact > 0.01: rec['intensity'] = "Significant Impact"
            else: rec['intensity'] = "Secondary Factor"
            recommendations.append(rec)
    
    # Fallback if no specific match
    if not recommendations:
        recommendations.append({
            "title": "Broad Spectrum Mindfulness",
            "advice": "Practice mindfulness-based stress reduction (MBSR) protocols to build general resilience.",
            "icon": "🧘",
            "intensity": "Baseline"
        })
        
    return recommendations[:3]
