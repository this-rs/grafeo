/// Detect if a query is about the system itself (identity, memory, facts).
/// These queries should be answered from persona facts + conv history,
/// NOT from the knowledge graph.
pub fn is_meta_query(query: &str) -> bool {
    let lower = query.to_lowercase();

    // Identity questions
    let identity_patterns = [
        "qui es-tu", "qui es tu", "c'est quoi ton nom", "quel est ton nom",
        "comment tu t'appelles", "comment t'appelles-tu",
        "what is your name", "who are you", "what are you",
        "tu es qui", "tu t'appelles comment",
    ];
    if identity_patterns.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Memory / fact recall questions
    let memory_patterns = [
        "qu'est-ce que tu sais sur moi", "que sais-tu de moi", "que sais-tu sur moi",
        "qu'est-ce que tu retiens", "que retiens-tu",
        "what do you know about me", "what do you remember",
        "tu te souviens", "te rappelles-tu", "te souviens-tu",
        "do you remember",
    ];
    if memory_patterns.iter().any(|p| lower.contains(p)) {
        return true;
    }

    false
}
