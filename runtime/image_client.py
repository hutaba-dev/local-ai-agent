"""Client contract for the local image generation service."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass

import httpx


IMAGE_API_URL = os.getenv("IMAGE_API_URL", "http://127.0.0.1:8001").rstrip("/")
POSE_API_URL = os.getenv("POSE_API_URL", "http://127.0.0.1:8002").rstrip("/")
IMAGE_WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").rstrip("/")
IMAGE_WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.8-27b")
ORIGINAL_RESTORE_PATTERN = re.compile(
    r"원래|되돌|복원|처음\s*(?:사진|이미지|얼굴|모습)|"
    r"\b(?:original|restore|revert|reset)\b",
    re.IGNORECASE,
)

ACTION_PROFILES = {
    "skateboarding": {
        "aliases": ("skateboard", "스케이트보드", "스케이트 보드"),
        "guidance": "full body; entire board visible; readable foot placement; believable center of gravity; clearly riding rather than standing",
    },
    "cycling": {
        "aliases": ("bicycle", "bike", "자전거"),
        "guidance": "full rider and bicycle visible; hands on handlebars; feet aligned with pedals; readable seated balance",
    },
    "running": {
        "aliases": ("running", "run", "달리", "뛰"),
        "guidance": "full body; clear stride phase; grounded contact or believable airborne phase; coordinated arm swing",
    },
    "dancing": {
        "aliases": ("dancing", "dance", "춤"),
        "guidance": "full body; expressive but anatomically coherent limb placement; readable rhythm and silhouette",
    },
    "jumping": {
        "aliases": ("jumping", "jump", "점프"),
        "guidance": "full body inside frame; believable lift and landing intent; coherent limb positions; visible ground context",
    },
    "sitting_at_desk": {
        "aliases": ("sitting at a desk", "desk", "책상", "앉"),
        "guidance": "readable seated posture; chair and desk relationship correct; hands interact naturally with the task object",
    },
    "walking": {
        "aliases": ("walking", "walk", "걷"),
        "guidance": "full or three-quarter body; natural gait; grounded feet; clear direction of travel",
    },
}

STYLE_PROFILES = {
    "anime_illustration": {
        "aliases": ("anime", "애니", "일본 애니"),
        "guidance": "clean linework; polished cel or soft shading; expressive coherent eyes; appealing character design; controlled background detail",
    },
    "semi_realistic_anime": {
        "aliases": ("semi-realistic anime", "반실사 애니"),
        "guidance": "anime design with natural facial proportions; dimensional soft shading; restrained expressive features",
    },
    "manga_lineart": {
        "aliases": ("manga", "만화 선화", "망가"),
        "guidance": "confident clean ink lines; controlled hatching; high silhouette readability; limited or monochrome color",
    },
    "cinematic_illustration": {
        "aliases": ("cinematic", "시네마틱"),
        "guidance": "intentional cinematic lighting; depth staging; coherent atmosphere; strong focal hierarchy",
    },
    "watercolor": {
        "aliases": ("watercolor", "수채화"),
        "guidance": "controlled translucent washes; clean focal features; soft edges away from the subject; harmonious color bleeding",
    },
    "3d_render": {
        "aliases": ("3d render", "3d 렌더", "3D"),
        "guidance": "coherent materials; physically plausible lighting; clean geometry; polished character rendering",
    },
    "photorealistic": {
        "aliases": ("photorealistic", "realistic photo", "실사", "사진처럼"),
        "guidance": "natural skin texture; realistic lens and lighting; plausible anatomy; subtle color grading",
    },
}

FEEDBACK_PATTERNS = {
    "face_failure": re.compile(r"얼굴|못생|face|eyes?|nose|mouth", re.IGNORECASE),
    "anatomy_failure": re.compile(r"인체|비율|팔다리|손|발|anatom|limb|hand|feet|foot", re.IGNORECASE),
    "pose_failure": re.compile(r"포즈|자세|동작|안\s*타|pose|action|not\s+(?:riding|running|walking)", re.IGNORECASE),
    "object_failure": re.compile(r"보드|자전거|물체|소품|object|missing", re.IGNORECASE),
    "composition_failure": re.compile(r"구도|잘렸|프레임|composition|cropped|framing", re.IGNORECASE),
    "style_failure": re.compile(r"스타일|화풍|느낌|style", re.IGNORECASE),
}
REFERENCE_RESEARCH_PATTERN = re.compile(
    r"검색.*참고|참고.*검색|레퍼런스|reference|look\s*up|visual\s*research",
    re.IGNORECASE,
)
EXPLICIT_PREFERENCE_PATTERN = re.compile(r"앞으로|항상|선호|취향|좋아해|싫어해|\b(?:prefer|always|my\s+style|dislike)\b", re.IGNORECASE)
VISUAL_PREFERENCE_PATTERN = re.compile(
    r"이미지|그림|얼굴|인체|구도|전신|스타일|화풍|애니|실사|색감|조명|"
    r"image|illustration|face|anatomy|composition|full.?body|style|anime|color|lighting",
    re.IGNORECASE,
)
PERSON_SUBJECT_PATTERN = re.compile(
    r"사람|인물|여자|남자|학생|소녀|소년|woman|man|person|girl|boy|student|character|portrait|초상",
    re.IGNORECASE,
)
AESTHETIC_INTENT_PATTERN = re.compile(
    r"예쁜|아름다운|미소녀|잘생긴|고급|매력|미감|attractive|beautiful|pretty|handsome|elegant|appealing",
    re.IGNORECASE,
)
IDENTITY_REPLACEMENT_PATTERN = re.compile(
    r"완전히\s*다른\s*사람|다른\s*사람처럼|replace\s+(?:the\s+)?(?:person|identity)|different\s+person",
    re.IGNORECASE,
)
EDIT_PATTERNS = {
    "appearance_refinement": re.compile(r"잘생기|예쁘게|아름답게|매력적|handsome|prettier|beautiful|attractive", re.IGNORECASE),
    "hair_edit": re.compile(r"머리|헤어|hair", re.IGNORECASE),
    "clothing_edit": re.compile(r"옷|의상|복장|clothing|clothes|outfit|wardrobe", re.IGNORECASE),
    "background_edit": re.compile(r"배경|background", re.IGNORECASE),
    "style_edit": re.compile(r"스타일|화풍|style|rendering", re.IGNORECASE),
    "lighting_edit": re.compile(r"조명|밝기|어둡|밝게|lighting|brighter|darker", re.IGNORECASE),
    "object_edit": re.compile(r"물체|소품|추가|제거|지워|object|add|remove", re.IGNORECASE),
    "expression_edit": re.compile(r"웃|미소|표정|smil|expression|frown", re.IGNORECASE),
    "face_orientation": re.compile(
        r"정면|바라보|쳐다보|고개|얼굴.*(?:왼쪽|오른쪽|돌)|(?:왼쪽|오른쪽).*얼굴|"
        r"front.?facing|face.*(?:left|right|camera)|head.*(?:left|right|turn)|gaze",
        re.IGNORECASE,
    ),
}
IDENTITY_CONSTRAINTS = (
    "preserve recognizable identity",
    "preserve ethnicity and approximate age",
    "preserve hairstyle unless requested",
    "use natural skin texture",
    "do not replace the face with a generic attractive face",
)


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    seed: int
    mode: str
    filename: str


@dataclass(frozen=True)
class ImagePromptPlan:
    prompt: str
    subject: str
    action: str
    style: str
    face_priority: str
    anatomy_priority: str
    must_have_object: str
    emotion: str = "unspecified"
    composition: str = "unspecified"
    camera: str = "unspecified"
    lighting: str = "unspecified"
    background: str = "unspecified"
    subject_design: str = "unspecified"
    hair: str = "unspecified"
    wardrobe: str = "unspecified"
    expression: str = "unspecified"
    color_palette: str = "unspecified"
    creative_brief: str = ""
    aesthetic_constraints: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    action_profile: str = "generic"
    style_profile: str = "generic"
    quality_sensitive: bool = False
    reference_research: bool = False


@dataclass(frozen=True)
class ImageQualityAssessment:
    checked: bool
    passed: bool
    failures: tuple[str, ...] = ()
    summary: str = ""
    scores: tuple[tuple[str, int], ...] = ()
    overall_score: float = 0.0
    decision: str = "accept"


@dataclass(frozen=True)
class ImageEditOperation:
    type: str
    target: str
    instruction: str
    capability: str


@dataclass(frozen=True)
class ImageEditPlan:
    edits: tuple[ImageEditOperation, ...]
    preserve_identity: bool = True
    constraints: tuple[str, ...] = IDENTITY_CONSTRAINTS


@dataclass(frozen=True)
class ImageEditCompletion:
    checked: bool
    passed: bool
    edit_status: tuple[tuple[str, bool], ...] = ()
    identity_preserved: bool = True
    summary: str = ""
    identity_score: int = 10


def _edit_instruction(edit_type: str, request: str) -> tuple[str, str, str]:
    if edit_type == "appearance_refinement":
        return (
            "face",
            "Naturally refine facial attractiveness with cleaner symmetry, balanced proportions, a clearer jawline, "
            "natural skin texture, and a flattering realistic structure while preserving recognizable identity.",
            "generative_edit",
        )
    if edit_type == "hair_edit":
        return "hair", f"Apply the requested hair change: {request}", "generative_edit"
    if edit_type == "clothing_edit":
        return "clothing", f"Apply the requested clothing change: {request}", "generative_edit"
    if edit_type == "background_edit":
        return "background", f"Apply the requested background change: {request}", "generative_edit"
    if edit_type == "style_edit":
        return "style", f"Apply the requested visual style change: {request}", "generative_edit"
    if edit_type == "lighting_edit":
        return "lighting", f"Apply the requested lighting change: {request}", "generative_edit"
    if edit_type == "object_edit":
        return "object", f"Apply the requested object change: {request}", "generative_edit"
    if edit_type == "expression_edit":
        return "facial_expression", f"Apply the requested expression change: {request}", "generative_edit"
    if edit_type == "generic_edit":
        return "image", f"Apply this requested image change exactly: {request}", "generative_edit"
    frontal = bool(re.search(r"정면|바라보|쳐다보|front.?facing|camera", request, re.IGNORECASE))
    return (
        "face_orientation",
        "Turn the face toward the camera while preserving identity." if frontal else f"Apply the requested face direction: {request}",
        "pose_correction" if frontal else "generative_edit",
    )


def build_image_edit_plan(request: str) -> ImageEditPlan:
    detected = [edit_type for edit_type, pattern in EDIT_PATTERNS.items() if pattern.search(request)]
    preserve_identity = not IDENTITY_REPLACEMENT_PATTERN.search(request)
    model_edits: list[dict[str, object]] = []
    instruction = """Extract every requested modification from an existing-image edit request.
Return one JSON object with an edits array. Each edit has type, target, instruction, and evidence, where evidence is an exact short quote from the user request. Allowed types: appearance_refinement, hair_edit, clothing_edit, background_edit, style_edit, lighting_edit, object_edit, expression_edit, face_orientation. Never collapse multiple modifications into one and never invent an unrequested change. Also return preserve_identity as a boolean. Do not include reasoning."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": request},
                ],
                "temperature": 0,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        model_edits = payload.get("edits", []) if isinstance(payload.get("edits"), list) else []
        preserve_identity = preserve_identity and bool(payload.get("preserve_identity", True))
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    ordered_types: list[str] = []
    allowed_types = set(EDIT_PATTERNS)
    for item in model_edits:
        edit_type = str(item.get("type", "")) if isinstance(item, dict) else ""
        evidence = str(item.get("evidence", "")).strip() if isinstance(item, dict) else ""
        if (
            edit_type in allowed_types
            and (EDIT_PATTERNS[edit_type].search(request) or evidence and evidence in request)
            and edit_type not in ordered_types
        ):
            ordered_types.append(edit_type)
    for edit_type in detected:
        if edit_type not in ordered_types:
            ordered_types.append(edit_type)
    if not ordered_types:
        ordered_types.append("generic_edit")
    operations = []
    for edit_type in ordered_types:
        target, edit_instruction, capability = _edit_instruction(edit_type, request)
        operations.append(ImageEditOperation(edit_type, target, edit_instruction, capability))
    if len(operations) > 1:
        operations.sort(key=lambda operation: operation.capability != "pose_correction")
    constraints = IDENTITY_CONSTRAINTS if preserve_identity else ("follow the requested identity replacement",)
    return ImageEditPlan(tuple(operations), preserve_identity, constraints)


def edit_plan_prompt(plan: ImageEditPlan) -> str:
    changes = " ".join(f"{index}. {edit.instruction}" for index, edit in enumerate(plan.edits, 1))
    constraints = "; ".join(plan.constraints)
    return f"Apply all requested edits cumulatively. {changes} Constraints: {constraints}. Preserve all completed prior edits."


def assess_image_edit_completion(
    plan: ImageEditPlan,
    original_image: bytes,
    final_image: bytes,
) -> ImageEditCompletion:
    requested = [{"type": edit.type, "target": edit.target, "instruction": edit.instruction} for edit in plan.edits]
    instruction = """Compare the original and final images and verify every requested edit independently.
Return JSON with edit_status as an object mapping each requested edit type to a boolean, identity_preserved as a boolean, identity_similarity as an integer from 0 to 10, and a short summary. Identity preservation requires the same recognizable person, ethnicity, approximate age, and core facial geometry; ordinary attractiveness improvement is not permission to replace or substantially rejuvenate the person. Do not provide reasoning."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\nRequested edits: {json.dumps(requested)}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(original_image).decode()}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(final_image).decode()}"}},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        raw_status = payload.get("edit_status", {})
        status = tuple((edit.type, bool(raw_status.get(edit.type, False))) for edit in plan.edits)
        identity_preserved = bool(payload.get("identity_preserved", False))
        identity_score = max(0, min(10, int(payload.get("identity_similarity", 0))))
        identity_passed = identity_preserved and identity_score >= 7
        passed = all(completed for _edit_type, completed in status) and (identity_passed or not plan.preserve_identity)
        return ImageEditCompletion(
            True, passed, status, identity_passed, str(payload.get("summary", ""))[:300], identity_score
        )
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return ImageEditCompletion(False, True, tuple((edit.type, True) for edit in plan.edits), True, "completion check unavailable")


def feedback_failure_labels(feedback: str) -> tuple[str, ...]:
    return tuple(label for label, pattern in FEEDBACK_PATTERNS.items() if pattern.search(feedback))


def requests_reference_research(request: str) -> bool:
    return bool(REFERENCE_RESEARCH_PATTERN.search(request))


def is_explicit_visual_preference(request: str) -> bool:
    return bool(EXPLICIT_PREFERENCE_PATTERN.search(request) and VISUAL_PREFERENCE_PATTERN.search(request))


def is_quality_sensitive_request(request: str, action_profile: str, style_profile: str) -> bool:
    person = bool(PERSON_SUBJECT_PATTERN.search(request))
    aesthetic = bool(AESTHETIC_INTENT_PATTERN.search(request))
    portrait = bool(re.search(r"portrait|초상|얼굴\s*(?:클로즈업|사진)", request, re.IGNORECASE))
    styled_action = person and action_profile != "generic" and style_profile != "generic"
    return portrait or (person and aesthetic) or styled_action


def analyze_visual_references(
    request: str,
    images: tuple[tuple[str, str, bytes], ...],
) -> str:
    """Extract reusable art-direction cues without copying a depicted identity or character."""
    if not images:
        return ""
    content: list[dict[str, object]] = [{
        "type": "text",
        "text": (
            "Analyze these visual references only for reusable pose, composition, camera, color mood, linework, "
            "facial-rendering, wardrobe-category, and background-density cues that suit the user request. "
            "Do not identify or reproduce any person, artist, character, logo, or copyrighted design. "
            f"Return a concise art-direction paragraph. User request: {request}"
        ),
    }]
    for _label, media_type, image_content in images[:3]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{base64.b64encode(image_content).decode()}"},
        })
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 260,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=90,
        )
        response.raise_for_status()
        cues = response.json()["choices"][0]["message"]["content"]
        return str(cues).strip()[:2_000]
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return ""


def _profile_for(text: str, profiles: dict[str, dict[str, object]]) -> tuple[str, str]:
    lowered = text.lower()
    for name, profile in profiles.items():
        aliases = profile["aliases"]
        if any(str(alias).lower() in lowered for alias in aliases):
            return name, str(profile["guidance"])
    return "generic", ""


def build_image_prompt(
    request: str,
    *,
    editing: bool = False,
    original_request: str = "",
    quality_feedback: str = "",
    simplify_composition: bool = False,
    preference_context: str = "",
    reference_cues: str = "",
) -> ImagePromptPlan:
    instruction = """Build a high-quality diffusion prompt from the user's image request.
Return one JSON object. String fields: prompt, subject, action, style, quality_intent, emotion, composition, camera, lighting, background, subject_design, hair, wardrobe, expression, color_palette, face_priority, anatomy_priority, must_have_object, creative_brief. Array fields: aesthetic_constraints, avoid. Boolean fields: quality_sensitive, reference_research.
The prompt must be concise English and prioritize, in order: subject/action correctness, anatomy, face quality, then style fidelity.
Include an appropriate composition and camera/view, background, all must-have elements, and explicit failure constraints.
When the user leaves visual choices open, synthesize one coherent creative brief and choose compatible subject design, hair, wardrobe, expression, camera, background density, lighting, and color palette instead of leaving them vague.
For a person performing an action, show the full body when needed, keep the core object fully visible, make limb placement and balance physically readable, keep the face unobscured unless requested, and use a clear silhouette.
For requested attractive or prominent people, require clear facial structure, symmetrical aligned eyes, a defined natural nose and mouth, and clean facial line work.
Combine a requested style with polished rendering, coherent anatomy, and character-art quality. Avoid distorted faces, broken anatomy, extra limbs, melted hands, disfigured feet, unclear poses, missing core objects, and floating bodies.
Do not imitate a living artist. Generic named media styles such as Japanese anime are allowed."""
    if editing:
        instruction += """
This is a localized edit. The prompt must explicitly say what to keep, what to change, and which quality issue to fix. Preserve successful identity, pose, composition, core objects, background, and style unless the user explicitly changes them."""
    else:
        instruction += """
This is generation from scratch. Do not refer to or depend on a previous image."""
    if simplify_composition:
        instruction += """
This is a retry after a failed result. Simplify the composition and camera angle while preserving the requested subject, action, core object, and style. Prefer a readable side or three-quarter view over a busy action shot."""
    user_content = (
        f"Original request: {original_request or request}\n"
        f"Current request or feedback: {request}\n"
        f"Known quality failures: {quality_feedback or 'none'}\n"
        f"Non-sensitive user visual preferences: {preference_context or 'none'}\n"
        f"Optional reference cues: {reference_cues or 'none'}"
    )
    action_profile, action_guidance = _profile_for(original_request or request, ACTION_PROFILES)
    style_profile, style_guidance = _profile_for(original_request or request, STYLE_PROFILES)
    profile_context = f"\nAction profile ({action_profile}): {action_guidance or 'generic clarity rules'}\nStyle profile ({style_profile}): {style_guidance or 'follow the requested medium consistently'}"
    user_content += profile_context
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_tokens": 900,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        prompt = str(payload.get("prompt", "")).strip()
        if prompt:
            return ImagePromptPlan(
                prompt=prompt[:2_000],
                subject=str(payload.get("subject", "unspecified"))[:160],
                action=str(payload.get("action", "none"))[:160],
                style=str(payload.get("style", "unspecified"))[:160],
                face_priority=str(payload.get("face_priority", "normal"))[:80],
                anatomy_priority=str(payload.get("anatomy_priority", "normal"))[:80],
                must_have_object=str(payload.get("must_have_object", "none"))[:160],
                emotion=str(payload.get("emotion", "unspecified"))[:120],
                composition=str(payload.get("composition", "unspecified"))[:240],
                camera=str(payload.get("camera", "unspecified"))[:160],
                lighting=str(payload.get("lighting", "unspecified"))[:160],
                background=str(payload.get("background", "unspecified"))[:240],
                subject_design=str(payload.get("subject_design", "unspecified"))[:240],
                hair=str(payload.get("hair", "unspecified"))[:160],
                wardrobe=str(payload.get("wardrobe", "unspecified"))[:240],
                expression=str(payload.get("expression", "unspecified"))[:160],
                color_palette=str(payload.get("color_palette", "unspecified"))[:160],
                creative_brief=str(payload.get("creative_brief", ""))[:600],
                aesthetic_constraints=tuple(str(item)[:160] for item in payload.get("aesthetic_constraints", [])[:12]),
                avoid=tuple(str(item)[:160] for item in payload.get("avoid", [])[:16]),
                action_profile=action_profile,
                style_profile=style_profile,
                quality_sensitive=bool(payload.get("quality_sensitive", False)) or is_quality_sensitive_request(
                    original_request or request, action_profile, style_profile
                ),
                reference_research=bool(payload.get("reference_research", False)),
            )
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    fallback = original_request or request
    if quality_feedback and quality_feedback != request:
        fallback = f"{fallback}. Correct these failures: {quality_feedback}"
    fallback += (
        ". Clear readable composition and camera view. Show the complete subject and every core object. "
        "For people in action, show a readable full-body pose with coherent balance and limb placement. "
        "Clear symmetrical face, natural aligned eyes, defined nose and mouth, coherent anatomy, polished rendering. "
        "Avoid distorted faces, broken anatomy, extra limbs, melted hands, disfigured feet, unclear action, "
        "missing objects, or floating bodies."
    )
    return ImagePromptPlan(
        fallback[:2_000], "unspecified", "unspecified", "unspecified", "high", "high", "unspecified",
        creative_brief="Fill unspecified design choices with one coherent, appealing visual concept.",
        aesthetic_constraints=("facial coherence", "appealing proportions", "polished design"),
        avoid=("distorted face", "broken anatomy", "unclear action", "missing core object"),
        action_profile=action_profile,
        style_profile=style_profile,
        quality_sensitive=is_quality_sensitive_request(original_request or request, action_profile, style_profile),
    )


def assess_image_quality(request: str, image_content: bytes) -> ImageQualityAssessment:
    instruction = """Evaluate whether the generated image satisfies the request. Inspect exactly these criteria:
subject present, requested action clearly readable, face coherent and aesthetically appropriate when visible, anatomy acceptable, main object present and readable, requested style roughly correct, and overall visual appeal.
Return one JSON object with integer scores from 0 to 10 for subject, action, face, anatomy, main_object, style, overall_appeal and a short summary string.
Use score 10 for a criterion that is genuinely not applicable. Be strict about severe visible failures, but do not reject harmless artistic variation."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\nUser request: {request}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_content).decode()}"}},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        criteria = ("subject", "action", "face", "anatomy", "main_object", "style", "overall_appeal")
        scores = tuple(
            (
                criterion,
                max(0, min(10, int(
                    payload.get(criterion, 10)
                    if not isinstance(payload.get(criterion), bool)
                    else (10 if payload.get(criterion) else 0)
                ))),
            )
            for criterion in criteria
        )
        score_map = dict(scores)
        critical = ("subject", "action", "face", "anatomy", "main_object", "style", "overall_appeal")
        failures = tuple(criterion for criterion in criteria if score_map[criterion] <= 3)
        weak_critical = sum(score_map[criterion] < 5 for criterion in critical)
        passed = not any(score_map[criterion] <= 3 for criterion in critical) and weak_critical < 2
        overall_score = round(sum(score_map.values()) / len(score_map), 2)
        return ImageQualityAssessment(
            True, passed, failures, str(payload.get("summary", ""))[:300], scores, overall_score,
            "accept" if passed else "regenerate",
        )
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return ImageQualityAssessment(False, True, (), "quality gate unavailable")


def create_image(prompt: str, source_image: bytes | None = None, strength: float = 0.5) -> GeneratedImage:
    payload: dict[str, object] = {"prompt": prompt, "strength": strength}
    if source_image is not None:
        payload["source_image_base64"] = base64.b64encode(source_image).decode()
    if IMAGE_WORKER_URL:
        capability = "image.edit" if source_image is not None else "image.generate"
        response = _worker_post(capability, payload)
    else:
        response = httpx.post(f"{IMAGE_API_URL}/v1/image", json=payload, timeout=300)
    response.raise_for_status()
    seed = int(response.headers["X-Image-Seed"])
    mode = response.headers["X-Image-Mode"]
    return GeneratedImage(response.content, seed, mode, f"{mode}-{seed}.png")


def correct_portrait_pose(source_image: bytes) -> GeneratedImage:
    payload = {"source_image_base64": base64.b64encode(source_image).decode()}
    if IMAGE_WORKER_URL:
        response = _worker_post("portrait.frontalize", payload)
    else:
        response = httpx.post(f"{POSE_API_URL}/v1/frontalize", json=payload, timeout=300)
    response.raise_for_status()
    return GeneratedImage(response.content, 0, "pose", "pose-corrected.png")


def _worker_post(capability: str, payload: dict[str, object]) -> httpx.Response:
    if not IMAGE_WORKER_TOKEN:
        raise RuntimeError("IMAGE_WORKER_TOKEN is required when IMAGE_WORKER_URL is configured")
    return httpx.post(
        f"{IMAGE_WORKER_URL}/v1/tasks/{capability}",
        json=payload,
        headers={"Authorization": f"Bearer {IMAGE_WORKER_TOKEN}"},
        timeout=300,
    )


def parse_image_command(message: str, source_image_available: bool = False) -> tuple[str, str] | None:
    stripped_message = message.strip()
    command, separator, prompt = stripped_message.partition(" ")
    if command not in {"/image", "/edit"}:
        intent = infer_image_intent(stripped_message, source_image_available)
        return None if intent == "chat" else (intent, stripped_message)
    if not separator or not prompt.strip():
        raise ValueError(f"{command} 뒤에 프롬프트를 입력하세요.")
    return command[1:], prompt.strip()


def prefers_original_source(message: str) -> bool:
    return bool(ORIGINAL_RESTORE_PATTERN.search(message))


def infer_image_intent(message: str, source_image_available: bool = False) -> str:
    instruction = """Determine the user's present intent in a conversation that can generate and edit images.
Decide from the meaning, discourse context, and speech act rather than matching keywords.
Return exactly one lowercase label:
- image: a present request to create or draw a new image from a description.
- edit: a small, localized change where the existing subject, identity, composition, pose, or background should be preserved.
- regenerate: a request to try again from scratch because the result has a severe quality failure, such as a distorted face, broken anatomy, unreadable action, wrong composition, missing core object, major style mismatch, or increasing edit drift.
- pose: a present request whose only modification is the subject's head direction, gaze, or front-facing pose. If orientation is combined with appearance, hair, clothing, expression, background, or style changes, choose edit.
- resend: a present request to display or send the existing image again without changing it.
- chat: all ordinary conversation, including commentary, feedback, future preferences, explanations, questions, and acknowledgements.
Treat direct complaints such as "the face is broken", "this looks wrong", "what did you draw", or "the person is not riding it" as requests to fix a severe failure now and choose regenerate when a source image is available. Do not use regenerate for a clearly localized correction that should preserve a successful image. Editing, regeneration, pose correction, and resending require an available source image; otherwise choose chat. If the intent is ambiguous, choose chat."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": f"Source image available: {'yes' if source_image_available else 'no'}\nUser message: {message}",
                    },
                ],
                "temperature": 0,
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        label = str(content).strip().lower().rstrip(".")
        allowed = {"image", "chat"}
        if source_image_available:
            allowed.update({"edit", "regenerate", "pose", "resend"})
        return label if label in allowed else "chat"
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return "chat"
