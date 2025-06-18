import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Literal, Annotated
from typing import List, Optional, Union
import validation_helper
import ast
import os

class Point(BaseModel):
    x: int  # Inclusive
    y: int  # Inclusive

    @model_validator(mode='after')
    def validate_data(self) -> 'Point':
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Point has negative coordinates: ({self.x}, {self.y})")
        return self

class BoundingBox(BaseModel):
    left: int  # Inclusive
    top: int  # Inclusive
    right: int  # Exclusive
    bottom: int  # Exclusive

    @model_validator(mode='after')
    def validate_data(self) -> 'BoundingBox':
        # Ensure positive dimensions
        if self.left < 0 or self.top < 0 or self.right < 0 or self.bottom < 0:
            raise ValueError(f"Bounding box has negative coordinates: {self}")

        # Ensure width and height make sense
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"Bounding box has invalid dimensions (non-positive area): {self}")

        width = self.right - self.left
        height = self.bottom - self.top

        # Warn for suspiciously small boxes
        if width < 5 or height < 5:
            warnings.warn(f"Very small bounding box: {width}x{height} at ({self.left}, {self.top})")

        # Warn for extreme aspect ratios
        # We use 10x to avoid false positives for real UI elements like search bars
        if width > 10 * height or height > 10 * width:
            warnings.warn(f"Unusual aspect ratio: {width}x{height} in {self}")
            
        # # Normalized bounds (should be in [0,1])
        # if max(self.left, self.top, self.right, self.bottom) > 1:
        #     warnings.warn(f"[Warning] Bounding box exceeds normalized range: {self}")
            
        return self

class UIElement(BaseModel):
    bbox: BoundingBox
    element_type: Optional[str]
    _instruction: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode='after')
    def validate_data(self) -> 'UIElement':
        if self.element_type and self._instruction:
            validation_helper.update_element_type_info(self.element_type, self._instruction)

        if not self.element_type:
            warnings.warn(f"[UIElement] Missing element_type: {self}")

        if not self._instruction or not self._instruction.strip():
            warnings.warn(f"[UIElement] Missing or empty instruction: {self}")
        return self

class UIAction(BaseModel):
    pyautogui: Optional[str] = None
    target_element: Optional[UIElement] = None
    target_point: Optional[Point] = None
    text_observation_desc: Optional[str] = None
    text_reasoning: Optional[str] = None
    text_subtask: Optional[str] = None
    # text_others: Optional[str] = None
    text_others: Optional[str] = None
    
    _stage: Optional[str] = PrivateAttr(default=None)

    def set_stage(self, stage: str):
        self._stage = stage
        return self


    @model_validator(mode='after')
    def validate_data(self) -> 'UIAction':
        if self.pyautogui:
            if not validation_helper.validate_pyautogui_command(self.pyautogui):
                warnings.warn(f"[UIAction] Invalid pyautogui command: {self.pyautogui}")
                
        # Handle stage 1/2 differences
        if self._stage == "stage1":
            for field in ["text_observation_desc", "text_reasoning", "text_subtask", "text_others"]:
                if getattr(self, field):
                    warnings.warn(f"[Stage 1] Unexpected field '{field}' present: {getattr(self, field)}")
                    
        elif self._stage == "stage2":
            # text_observation_desc
            if self.text_observation_desc:
                if not self.text_observation_desc.strip():
                    warnings.warn("[UIAction] text_observation_desc is empty or whitespace only")
            elif self.pyautogui or self.text_subtask:
                warnings.warn("[UIAction] Missing text_observation_desc while action is present")

            # text_reasoning
            if self.text_reasoning:
                if not self.text_reasoning.strip():
                    warnings.warn("[UIAction] text_reasoning is empty")
            elif self.pyautogui or self.text_subtask:
                warnings.warn("[UIAction] Missing text_reasoning while action is present")

            # text_subtask
            if self.text_subtask:
                if not self.text_subtask.strip():
                    warnings.warn("[UIAction] text_subtask is empty")
                elif len(self.text_subtask.strip().split()) <= 2:
                    warnings.warn(f"[UIAction] text_subtask may be too vague: {self.text_subtask}")
                elif not self.text_subtask.lower().startswith("action"):
                    warnings.warn(f"[UIAction] text_subtask does not follow 'Action:' style: {self.text_subtask}")
            elif self.pyautogui:
                warnings.warn("[UIAction] Missing text_subtask while pyautogui action is present")

            # text_others
            if self.text_others is not None and not isinstance(self.text_others, str):
                warnings.warn("[UIAction] text_others is not a string")
            elif self.text_others:
                redundancy = any(self.text_others in (self.text_reasoning or '', self.text_subtask or ''))
                if redundancy:
                    warnings.warn("[UIAction] text_others may be redundant with reasoning/subtask")

        return self

class UIConversationInput(BaseModel):
    from_: Annotated[Literal['human'], Field(alias='from')]
    value: str

    @model_validator(mode='after')
    def validate_data(self) -> 'UIConversationInput':
        if not self.value.strip():
            warnings.warn("[UIConversationInput] value is empty or whitespace")
        # elif len(self.value.strip().split()) < 4:
        #     warnings.warn(f"[UIConversationInput] value is very short: '{self.value}'")
        if "<image>" not in self.value:
            warnings.warn("[UIConversationInput] value does not contain '<image>' placeholder")
        # if not any(keyword in self.value.lower() for keyword in ["click", "select", "type"]):
        #     warnings.warn("[UIConversationInput] value might not contain clear instruction-related keywords")
        return self

class UIConversationOutput(BaseModel):
    from_: Annotated[Literal['gpt'], Field(alias='from')]
    actions: List[UIAction]
    
    _stage: Optional[str] = PrivateAttr(default=None)

    def set_stage(self, stage: str):
        self._stage = stage
        for action in self.actions:
            action.set_stage(stage)
        return self

    @model_validator(mode='after')
    def validate_data(self) -> 'UIConversationOutput':
        if not self.actions:
            warnings.warn("[UIConversationOutput] No actions provided")

        all_empty = True
        for a in self.actions:
            if a.pyautogui or a.text_subtask:
                all_empty = False
                break
        if len(self.actions) > 1 and all_empty:
            warnings.warn("[UIConversationOutput] Multiple actions present but none have pyautogui/text_subtask")

        return self

Conversation = List[Union[UIConversationInput, UIConversationOutput]]

class InteractionStep(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    all_ui_elements: Optional[List[UIElement]]
    conversation_list: List[Conversation]

    @model_validator(mode='after')
    def validate_data(self) -> 'InteractionStep':
        if not self.image_path.strip():
            raise ValueError("[InteractionStep] image_path is empty or whitespace")
        elif not self.image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError(f"[InteractionStep] image_path has unexpected format: {self.image_path}")
        # elif not os.path.exists(self.image_path):
        #     warnings.warn(f"[InteractionStep] image_path does not exist on disk: {self.image_path}")
        elif os.path.isabs(self.image_path):
            warnings.warn(f"[InteractionStep] image_path is absolute, expected relative: {self.image_path}")

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("[InteractionStep] Invalid image dimensions: {self.image_width}x{self.image_height}")
        elif self.image_width < 100 or self.image_height < 100:
            warnings.warn(f"[InteractionStep] image dimensions unusually small: {self.image_width}x{self.image_height}")
        elif self.image_width > 4000 or self.image_height > 4000:
            warnings.warn(f"[InteractionStep] image dimensions unusually large: {self.image_width}x{self.image_height}")

        if self.all_ui_elements is not None:
            if len(self.all_ui_elements) == 0:
                warnings.warn("[InteractionStep] all_ui_elements is provided but empty")
            elif len(self.all_ui_elements) > 1000:
                warnings.warn("[InteractionStep] all_ui_elements has too many entries — possible parse issue")

            seen_bboxes = set()
            for el in self.all_ui_elements:
                bbox = el.bbox
                width = bbox.right - bbox.left
                height = bbox.bottom - bbox.top

                abs_left = bbox.left
                abs_right = bbox.right
                abs_top = bbox.top
                abs_bottom = bbox.bottom

                # Bounds check
                if abs_left < 0 or abs_top < 0 or abs_right > self.image_width or abs_bottom > self.image_height:
                    warnings.warn(f"[InteractionStep] BoundingBox out of image bounds: {bbox} with image size ({self.image_width}x{self.image_height})")

                # Reasonable size
                if width > self.image_width or height > self.image_height:
                    warnings.warn(f"[InteractionStep] BBox exceeds image size: {bbox}")
                if width > 0.9 * self.image_width or height > 0.9 * self.image_height:
                    warnings.warn(f"[InteractionStep] BBox covers >90% of screen: {bbox}")

                # Border check
                if abs_left <= 2 or abs_top <= 2 or abs_right >= self.image_width - 2 or abs_bottom >= self.image_height - 2:
                    warnings.warn(f"[InteractionStep] BBox too close to screen edge: {bbox}")

                # Duplicate detection
                key = (bbox.left, bbox.top, bbox.right, bbox.bottom)
                if key in seen_bboxes:
                    warnings.warn(f"[InteractionStep] Duplicate bounding box detected: {bbox}")
                seen_bboxes.add(key)

        if not self.conversation_list or len(self.conversation_list) == 0:
            warnings.warn("[InteractionStep] conversation_list is empty")

        # Target point checks
        for convo in self.conversation_list or []:
            for turn in convo:
                if hasattr(turn, "actions"):
                    for action in turn.actions:
                        if action.target_point:
                            abs_x = action.target_point.x
                            abs_y = action.target_point.y
                            if abs_x > self.image_width or abs_y > self.image_height:
                                warnings.warn(f"[InteractionStep] target_point ({abs_x}, {abs_y}) exceeds image size ({self.image_width}x{self.image_height})")

        return self

class Trajectory(BaseModel):
    data_source: str
    is_navigation: bool
    domain: Literal['macos', 'linux', 'windows', 'web', 'mobile']
    steps: List[InteractionStep]
    
    @model_validator(mode='after')
    def validate_data(self) -> 'Trajectory':
        # === 1. data_source ===
        allowed_sources = {"aguvis-stage1", "aguvis-stage2", "os-atlas"}  # Add more as needed
        if not self.data_source.strip():
            warnings.warn("[Trajectory] data_source is empty or whitespace")
        elif self.data_source not in allowed_sources:
            warnings.warn(f"[Trajectory] Unknown data_source: {self.data_source}")

        # === 2. is_navigation ===
        if not isinstance(self.is_navigation, bool):
            warnings.warn(f"[Trajectory] is_navigation is not boolean: {self.is_navigation}")

        # === 3. domain ===
        allowed_domains = {"macos", "linux", "windows", "web", "mobile"}
        if self.domain not in allowed_domains:
            warnings.warn(f"[Trajectory] Invalid domain: {self.domain}")

        # === 4. steps ===
        if not self.steps:
            warnings.warn("[Trajectory] steps is empty")
        elif len(self.steps) < 2:
            warnings.warn(f"[Trajectory] steps has very few entries: {len(self.steps)}")
        elif len(self.steps) > 50:
            warnings.warn(f"[Trajectory] steps has unusually many entries: {len(self.steps)}")

        # for i, step in enumerate(self.steps):
        #     try:
        #         step.validate_data()  # Should already be called, but ensures each step is individually valid
        #     except ValidationError as e:
        #         warnings.warn(f"[Trajectory] Step {i} failed validation: {e}")

        # === Deferred to parser ===
        # - Check if all steps share the same domain (global consistency)
        # - Check if all image sizes are identical (resolution consistency)
        # - Detect duplicate image paths across steps
        # - Detect repeated pyautogui commands across steps (redundancy detection)
        # - Check domain consistency with image_path prefix

        return self
    
class ConversationBundle(BaseModel):
    conversation: Conversation

    @model_validator(mode="after")
    def validate_conversation(self) -> 'ConversationBundle':
        turns = self.conversation
        for i, turn in enumerate(turns):
            if i == 0 and isinstance(turn, UIConversationOutput):
                warnings.warn("[ConversationBundle] Starts with GPT — may be invalid unless intentional")

            if isinstance(turn, UIConversationInput):
                if i + 1 >= len(turns) or not isinstance(turns[i + 1], UIConversationOutput):
                    warnings.warn(f"[ConversationBundle] Human turn at index {i} is not followed by GPT turn")

            if i > 0 and isinstance(turn, type(turns[i - 1])):
                warnings.warn(f"[ConversationBundle] Two consecutive '{turn.from_}' turns at index {i-1}, {i}")

            # Stage propagation
            if isinstance(turn, UIConversationOutput):
                for action in turn.actions:
                    if action._stage is None and turn._stage:
                        action.set_stage(turn._stage)

        # Final turn check
        if isinstance(turns[-1], UIConversationInput):
            warnings.warn("[ConversationBundle] Ends with a human turn — GPT response may be missing")

        return self
