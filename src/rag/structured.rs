use super::{OnionSlice, OnionSliceConfig, SliceLayer, extract_keywords};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct SemanticBlock {
    pub role_heading: String,
    pub primary_label: &'static str,
    pub content: String,
    pub summary: String,
    pub facets: Vec<SemanticFacet>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticFacet {
    pub label: &'static str,
    pub text: String,
}

#[derive(Debug, Clone)]
struct RawBlock {
    role: String,
    content: String,
}

pub fn is_structured_conversation(metadata: &Value) -> bool {
    metadata.as_object().is_some_and(|object| {
        matches!(
            object.get("type").and_then(|value| value.as_str()),
            Some("conversation" | "transcript_turn")
        ) || matches!(
            object.get("format").and_then(|value| value.as_str()),
            Some("markdown_transcript" | "sessions" | "claude_web" | "chatgpt")
        )
    })
}

pub fn parse_blocks(content: &str, metadata: &Value) -> Vec<SemanticBlock> {
    let raw_blocks = if metadata
        .get("format")
        .and_then(|value| value.as_str())
        .is_some_and(|format| format == "markdown_transcript")
        || content
            .lines()
            .any(|line| parse_markdown_heading(line).is_some())
    {
        parse_markdown_transcript_blocks(content)
    } else {
        vec![RawBlock {
            role: metadata
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("message")
                .to_string(),
            content: content.trim().to_string(),
        }]
    };

    raw_blocks
        .into_iter()
        .filter_map(|block| {
            let content = block.content.trim();
            if content.is_empty() {
                return None;
            }

            let role_key = normalize_role_key(&block.role);
            let primary_label = primary_label(&role_key);
            let summary = summarize_text(content, 96);

            let mut facets = vec![SemanticFacet {
                label: primary_label,
                text: summary.clone(),
            }];

            if let Some(decision) = infer_decision(content) {
                facets.push(SemanticFacet {
                    label: "Decision",
                    text: decision,
                });
            }
            if let Some(next) = infer_next_action(content) {
                facets.push(SemanticFacet {
                    label: "Next",
                    text: next,
                });
            }
            if let Some(entities) = infer_entities(metadata, content) {
                facets.push(SemanticFacet {
                    label: "Entities",
                    text: entities,
                });
            }

            dedupe_facets(&mut facets);

            Some(SemanticBlock {
                role_heading: role_heading(&role_key, &block.role),
                primary_label,
                content: content.to_string(),
                summary,
                facets,
            })
        })
        .collect()
}

fn parse_markdown_transcript_blocks(content: &str) -> Vec<RawBlock> {
    let mut blocks = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_lines = Vec::new();

    for line in content.lines() {
        if let Some(role) = parse_markdown_heading(line) {
            if let Some(existing_role) = current_role.take() {
                push_raw_block(&mut blocks, existing_role, &current_lines.join("\n"));
            }
            current_role = Some(role.to_string());
            current_lines.clear();
            continue;
        }

        current_lines.push(line.to_string());
    }

    if let Some(existing_role) = current_role {
        push_raw_block(&mut blocks, existing_role, &current_lines.join("\n"));
    }

    if blocks.is_empty() {
        push_raw_block(&mut blocks, "transcript".to_string(), content);
    }

    blocks
}

fn push_raw_block(blocks: &mut Vec<RawBlock>, role: String, content: &str) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }

    blocks.push(RawBlock {
        role,
        content: trimmed.to_string(),
    });
}

fn parse_markdown_heading(line: &str) -> Option<&'static str> {
    let trimmed = line.trim();
    if trimmed.eq_ignore_ascii_case("User request:") {
        Some("user")
    } else if trimmed.eq_ignore_ascii_case("Assistant response:") {
        Some("assistant")
    } else if trimmed.eq_ignore_ascii_case("Reasoning focus:") {
        Some("reasoning")
    } else {
        None
    }
}

fn normalize_role_key(role: &str) -> String {
    match role.trim().to_ascii_lowercase().as_str() {
        "human" => "user".to_string(),
        "bot" => "assistant".to_string(),
        other => other.to_string(),
    }
}

fn primary_label(role: &str) -> &'static str {
    match role {
        "user" => "Request",
        "assistant" => "Response",
        "reasoning" => "Reasoning",
        "system" => "Context",
        "tool" => "Tool",
        _ => "Message",
    }
}

fn role_heading(role: &str, fallback: &str) -> String {
    match role {
        "user" => "User request".to_string(),
        "assistant" => "Assistant response".to_string(),
        "reasoning" => "Reasoning focus".to_string(),
        "system" => "System context".to_string(),
        "tool" => "Tool output".to_string(),
        _ => title_case(fallback),
    }
}

fn title_case(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return "Message".to_string();
    }

    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return "Message".to_string();
    };

    let mut result = first.to_uppercase().collect::<String>();
    result.push_str(chars.as_str());
    result
}

fn summarize_text(text: &str, max_chars: usize) -> String {
    let candidate = first_non_empty_line(text)
        .or_else(|| sentence_candidates(text).into_iter().next())
        .unwrap_or_else(|| collapse_whitespace(text));
    truncate_at_word_boundary(&candidate, max_chars)
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(collapse_whitespace)
}

fn sentence_candidates(text: &str) -> Vec<String> {
    let normalized = text.replace('\n', " ");
    normalized
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(collapse_whitespace)
        .collect()
}

const INLINE_SEMANTIC_LABELS: [&str; 9] = [
    "decision:",
    "decided:",
    "resolution:",
    "next action:",
    "next steps:",
    "next:",
    "todo:",
    "action item:",
    "follow-up:",
];

fn find_labeled_fragment(text: &str, labels: &[&str]) -> Option<String> {
    text.lines().map(str::trim).find_map(|line| {
        let lower = line.to_ascii_lowercase();
        labels.iter().find_map(|label| {
            let start = lower.find(label)?;
            let remainder = &line[start + label.len()..];
            let remainder_lower = remainder.to_ascii_lowercase();
            let cut_idx = INLINE_SEMANTIC_LABELS
                .iter()
                .filter_map(|other_label| remainder_lower.find(other_label))
                .min()
                .unwrap_or(remainder.len());
            let fragment = remainder[..cut_idx].trim();

            if fragment.is_empty() {
                None
            } else {
                Some(truncate_at_word_boundary(
                    &collapse_whitespace(fragment),
                    96,
                ))
            }
        })
    })
}

fn find_candidate_by_keywords(text: &str, keywords: &[&str]) -> Option<String> {
    text.lines()
        .chain(text.split(['.', '!', '?']))
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .find(|segment| {
            let lower = segment.to_ascii_lowercase();
            keywords.iter().any(|keyword| lower.contains(keyword))
        })
        .map(|segment| truncate_at_word_boundary(&collapse_whitespace(segment), 96))
}

fn infer_decision(text: &str) -> Option<String> {
    find_labeled_fragment(text, &["decision:", "decided:", "resolution:"]).or_else(|| {
        find_candidate_by_keywords(
            text,
            &[
                "decid",
                "agreed",
                "going with",
                "chosen",
                "we will use",
                "resolved",
            ],
        )
    })
}

fn infer_next_action(text: &str) -> Option<String> {
    find_labeled_fragment(
        text,
        &[
            "next action:",
            "next steps:",
            "next:",
            "todo:",
            "action item:",
            "follow-up:",
        ],
    )
    .or_else(|| {
        find_candidate_by_keywords(
            text,
            &[
                "next",
                "todo",
                "follow up",
                "follow-up",
                "need to",
                "i'll",
                "we'll",
                "will add",
                "will wire",
                "plan to",
            ],
        )
    })
}

fn infer_entities(metadata: &Value, content: &str) -> Option<String> {
    let mut entities = Vec::new();

    if let Some(object) = metadata.as_object() {
        for key in ["project", "title", "conversation", "session", "agent"] {
            if let Some(value) = object.get(key).and_then(|value| value.as_str()) {
                let trimmed = value.trim();
                if !trimmed.is_empty() && trimmed != "unknown" {
                    entities.push(trimmed.to_string());
                }
            }
        }
    }

    for keyword in extract_keywords(content, 6) {
        if keyword.len() > 3 {
            entities.push(keyword);
        }
    }

    entities.dedup();
    if entities.is_empty() {
        None
    } else {
        Some(truncate_at_word_boundary(&entities.join(", "), 96))
    }
}

fn dedupe_facets(facets: &mut Vec<SemanticFacet>) {
    let mut unique = Vec::with_capacity(facets.len());
    for facet in facets.drain(..) {
        let is_duplicate = unique.iter().any(|existing: &SemanticFacet| {
            existing.label == facet.label || existing.text.eq_ignore_ascii_case(&facet.text)
        });
        if !is_duplicate {
            unique.push(facet);
        }
    }
    *facets = unique;
}

fn collect_facets(blocks: &[SemanticBlock], labels: &[&'static str]) -> Vec<String> {
    let mut segments = Vec::new();
    for label in labels {
        for block in blocks {
            if let Some(facet) = block.facets.iter().find(|facet| facet.label == *label) {
                let segment = format!("{}: {}", facet.label, facet.text);
                if !segments.iter().any(|existing| existing == &segment) {
                    segments.push(segment);
                }
            }
        }
    }
    segments
}

fn pack_segments(segments: &[String], target_chars: usize) -> String {
    let mut result = String::new();

    for segment in segments {
        let candidate = if result.is_empty() {
            segment.clone()
        } else {
            format!("{result} | {segment}")
        };

        if candidate.chars().count() <= target_chars {
            result = candidate;
            continue;
        }

        if result.is_empty() {
            return truncate_at_word_boundary(segment, target_chars);
        }

        if result.chars().count() + 4 <= target_chars {
            result.push_str(" | …");
        }
        break;
    }

    if result.is_empty() {
        truncate_at_word_boundary(&segments.join(" | "), target_chars)
    } else {
        result
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_at_word_boundary(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let byte_idx = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let truncated = &text[..byte_idx];

    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &truncated[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

pub fn create_structured_outer(blocks: &[SemanticBlock], target_chars: usize) -> String {
    let mut segments: Vec<String> = blocks
        .iter()
        .map(|block| format!("{}: {}", block.primary_label, block.summary))
        .collect();
    segments.extend(collect_facets(blocks, &["Decision", "Next", "Entities"]));
    pack_segments(&segments, target_chars)
}

pub fn create_structured_middle(blocks: &[SemanticBlock], target_chars: usize) -> String {
    let mut sections = Vec::new();

    for block in blocks {
        sections.push(format!(
            "{}: {}",
            block.primary_label,
            truncate_at_word_boundary(
                &block.summary,
                match block.primary_label {
                    "Request" => 36,
                    "Response" => 44,
                    "Reasoning" => 40,
                    _ => 40,
                }
            )
        ));
    }

    for facet in collect_facets(blocks, &["Decision"]) {
        sections.push(truncate_at_word_boundary(&facet, 44));
    }
    for facet in collect_facets(blocks, &["Next"]) {
        sections.push(truncate_at_word_boundary(&facet, 44));
    }
    for facet in collect_facets(blocks, &["Entities"]) {
        sections.push(truncate_at_word_boundary(&facet, 36));
    }

    truncate_at_word_boundary(&sections.join("\n"), target_chars)
}

pub fn create_structured_inner(blocks: &[SemanticBlock], target_chars: usize) -> String {
    let chars_per_block = (target_chars / blocks.len().max(1)).max(120);
    let mut sections = Vec::new();

    for block in blocks {
        let excerpt =
            truncate_at_word_boundary(&collapse_whitespace(&block.content), chars_per_block);
        sections.push(format!("{}:\n{}", block.role_heading, excerpt));
    }

    let details = collect_facets(blocks, &["Decision", "Next", "Entities"]);
    if !details.is_empty() {
        sections.push(details.join("\n"));
    }

    truncate_at_word_boundary(&sections.join("\n\n"), target_chars)
}

pub fn create_structured_onion_slices(
    content: &str,
    metadata: &Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();
    let blocks = parse_blocks(content, metadata);

    if content.len() < config.min_content_for_slicing {
        return create_structured_outer_core_slices(content, &blocks, config);
    }

    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    let inner_content = create_structured_inner(&blocks, config.inner_target);
    let inner_id = OnionSlice::generate_id(&inner_content, SliceLayer::Inner);
    let inner_keywords = extract_keywords(&inner_content, 7);

    let middle_content = create_structured_middle(&blocks, config.middle_target);
    let middle_id = OnionSlice::generate_id(&middle_content, SliceLayer::Middle);
    let middle_keywords = extract_keywords(&middle_content, 5);

    let outer_content = create_structured_outer(&blocks, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(middle_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: middle_id.clone(),
            layer: SliceLayer::Middle,
            content: middle_content,
            parent_id: Some(inner_id.clone()),
            children_ids: vec![outer_id],
            keywords: middle_keywords,
        },
        OnionSlice {
            id: inner_id.clone(),
            layer: SliceLayer::Inner,
            content: inner_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![middle_id],
            keywords: inner_keywords,
        },
        OnionSlice {
            id: core_id.clone(),
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![inner_id],
            keywords: core_keywords,
        },
    ]
}

pub fn create_structured_onion_slices_fast(
    content: &str,
    metadata: &Value,
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let content = content.trim();
    let blocks = parse_blocks(content, metadata);

    if content.len() < config.min_content_for_slicing {
        return create_structured_outer_core_slices(content, &blocks, config);
    }

    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    let outer_content = create_structured_outer(&blocks, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![outer_id],
            keywords: core_keywords,
        },
    ]
}

fn create_structured_outer_core_slices(
    content: &str,
    blocks: &[SemanticBlock],
    config: &OnionSliceConfig,
) -> Vec<OnionSlice> {
    let core_id = OnionSlice::generate_id(content, SliceLayer::Core);
    let core_keywords = extract_keywords(content, 10);

    let outer_content = create_structured_outer(blocks, config.outer_target);
    let outer_id = OnionSlice::generate_id(&outer_content, SliceLayer::Outer);
    let outer_keywords = extract_keywords(&outer_content, 3);

    vec![
        OnionSlice {
            id: outer_id.clone(),
            layer: SliceLayer::Outer,
            content: outer_content,
            parent_id: Some(core_id.clone()),
            children_ids: vec![],
            keywords: outer_keywords,
        },
        OnionSlice {
            id: core_id,
            layer: SliceLayer::Core,
            content: content.to_string(),
            parent_id: None,
            children_ids: vec![outer_id],
            keywords: core_keywords,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::{create_structured_outer, parse_blocks};
    use serde_json::json;

    #[test]
    fn structured_outer_prefers_semantic_card_over_keyword_prefix() {
        let metadata = json!({
            "type": "conversation",
            "format": "claude_web",
            "role": "assistant",
            "title": "Pipeline progress",
            "project": "VetCoders/rmcp-memex"
        });
        let content = "Decision: use semantic cards for outer retrieval. Next action: add JSON regression tests and keep plain-text fallback.";

        let blocks = parse_blocks(content, &metadata);
        let outer = create_structured_outer(&blocks, 260);

        assert!(outer.contains("Response:"));
        assert!(outer.contains("Decision:"));
        assert!(outer.contains("Next:"));
        assert!(!outer.starts_with('['));
    }
}
