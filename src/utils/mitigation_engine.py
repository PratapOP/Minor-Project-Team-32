def get_mitigation_strategies(top_features):
    """
    Maps top stress factors to actionable mitigation strategies.
    """
    strategy_map = {
        "Have you recently experienced stress in your life?": {
            "title": "General Stress Management",
            "advice": "Practice 4-7-8 breathing exercises and prioritize task lists to reduce overwhelming feelings.",
            "icon": "🧘"
        },
        "Have you noticed a rapid heartbeat or palpitations?": {
            "title": "Physical Calmness",
            "advice": "Reduce caffeine intake and consider a 5-minute progressive muscle relaxation session.",
            "icon": "💓"
        },
        "Have you been dealing with anxiety or tension recently?": {
            "title": "Anxiety Reduction",
            "advice": "Try grounding techniques (5-4-3-2-1 method) and limit news/social media consumption.",
            "icon": "🛡️"
        },
        "Do you face any sleep problems or difficulties falling asleep?": {
            "title": "Sleep Hygiene",
            "advice": "Maintain a consistent sleep schedule and implement a 'Digital Detox' 1 hour before bed.",
            "icon": "🌙"
        },
        "Have you been getting headaches more often than usual?": {
            "title": "Physical Relief",
            "advice": "Ensure proper hydration and take eye-rest breaks (20-20-20 rule) if working on screens.",
            "icon": "💆"
        },
        "Do you get irritated easily?": {
            "title": "Emotional Regulation",
            "advice": "Identify triggers and practice 'Pause and Reflect' before responding to stressful situations.",
            "icon": "⚖️"
        },
        "Do you have trouble concentrating on your academic tasks?": {
            "title": "Focus Optimization",
            "advice": "Use the Pomodoro Technique (25m work, 5m break) and clear your workspace of distractions.",
            "icon": "🎯"
        },
        "Have you been feeling sadness or low mood?": {
            "title": "Mood Elevation",
            "advice": "Engage in light physical activity (e.g., a 15-minute walk) and connect with a trusted friend.",
            "icon": "☀️"
        },
        "Do you often feel lonely or isolated?": {
            "title": "Social Connection",
            "advice": "Join a student club or community group. Even small social interactions can significantly boost mood.",
            "icon": "🤝"
        },
        "Do you feel overwhelmed with your academic workload?": {
            "title": "Workload Management",
            "advice": "Break large projects into micro-tasks and seek guidance from academic advisors or mentors.",
            "icon": "📚"
        },
        "Are you in competition with your peers, and does it affect you?": {
            "title": "Perspective Shift",
            "advice": "Focus on personal growth rather than comparison. Collaborative study sessions can reduce competitive tension.",
            "icon": "🏁"
        },
        "Do you find that your relationship often causes you stress?": {
            "title": "Relational Boundaries",
            "advice": "Practice open communication with your partner and set healthy boundaries for personal time.",
            "icon": "💬"
        },
        "Are you facing any difficulties with your professors or instructors?": {
            "title": "Professional Communication",
            "advice": "Schedule formal office hours to discuss concerns calmly and seek clarification on expectations.",
            "icon": "🎓"
        },
        "Is your working environment unpleasant or stressful?": {
            "title": "Environmental Audit",
            "advice": "Personalize your workspace with plants or calming items. Consider noise-canceling headphones.",
            "icon": "🏢"
        },
        "Do you struggle to find time for relaxation and leisure activities?": {
            "title": "Time Auditing",
            "advice": "Schedule 'Rest Blocks' in your calendar just like academic tasks. Leisure is essential for productivity.",
            "icon": "🎡"
        },
        "Is your hostel or home environment causing you difficulties?": {
            "title": "Living Space Optimization",
            "advice": "Create a designated 'Zon-Out' space in your room. Discuss noise concerns with roommates respectfully.",
            "icon": "🏠"
        },
        "Do you lack confidence in your academic performance?": {
            "title": "Confidence Building",
            "advice": "Celebrate small wins. Keep a 'Success Journal' of things you've accomplished each week.",
            "icon": "💎"
        },
        "Academic and extracurricular activities conflicting for you?": {
            "title": "Priority Balancing",
            "advice": "Evaluate your commitments and consider temporarily reducing extracurricular load during peak exam times.",
            "icon": "🤹"
        }
    }

    recommendations = []
    for feat, impact in top_features:
        if feat in strategy_map:
            recommendations.append(strategy_map[feat])
    
    # Fallback if no specific match
    if not recommendations:
        recommendations.append({
            "title": "Mindfulness Protocol",
            "advice": "Consistent mindfulness and physical activity remain the gold standard for stress reduction.",
            "icon": "🧘"
        })
        
    return recommendations[:3] # Return top 3
