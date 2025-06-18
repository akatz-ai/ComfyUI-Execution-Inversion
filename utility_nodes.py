import comfy.samplers
import torch
from .tools import VariantSupport
from .base_node import NODE_POSTFIX, ListNode, DebugNode

VALID_SAMPLERS = comfy.samplers.KSampler.SAMPLERS
VALID_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

@VariantSupport()
class AccumulateNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"
    
    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

@VariantSupport()
class AccumulationHeadNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

@VariantSupport()
class AccumulationTailNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

@VariantSupport()
class AccumulationToListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "accumulation_to_list"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

@VariantSupport()
class ListToAccumulationNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = True
    FUNCTION = "list_to_accumulation"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

@VariantSupport()
class AccumulationGetLengthNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "accumlength"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)
        
@VariantSupport()
class AccumulationGetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "get_item"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)
        
@VariantSupport()
class AccumulationSetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "set_item"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

@VariantSupport()
class DebugPrint(DebugNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "label": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "debug_print"

    def debugtype(self, value):
        if isinstance(value, list):
            result = "["
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += "]"
        elif isinstance(value, tuple):
            result = "("
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += ")"
        elif isinstance(value, dict):
            result = "{"
            for k, v in value.items():
                result += ("%s: %s," % (self.debugtype(k), self.debugtype(v)))
            result += "}"
        elif isinstance(value, str):
            result = "'%s'" % value
        elif isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            result = str(value)
        elif isinstance(value, torch.Tensor):
            result = "Tensor[%s]" % str(value.shape)
        else:
            result = type(value).__name__
        return result

    def debug_print(self, value, label):
        print("[%s]: %s" % (label, self.debugtype(value)))
        return (value,)

NUM_LIST_SOCKETS = 10
@VariantSupport()
class MakeListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("*",),
            },
            "optional": {
                "value%d" % i: ("*",) for i in range(1, NUM_LIST_SOCKETS)
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "make_list"
    OUTPUT_IS_LIST = (True,)

    def make_list(self, **kwargs):
        result = []
        for i in range(NUM_LIST_SOCKETS):
            if "value%d" % i in kwargs:
                result.append(kwargs["value%d" % i])
        return (result,)

@VariantSupport()
class GetFloatFromList(ListNode):
    """
    Get the float from the list at the index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("FLOAT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float_from_list"
    
    def get_float_from_list(self, list: list, index: int):
        return (list[index],)
    
@VariantSupport()
class GetIntFromList(ListNode):
    """
    Get the int from the list at the index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("INT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int_from_list"
    
    def get_int_from_list(self, list: list, index: int):
        return (list[index],)

# Configuration for node display names

UTILITY_NODE_CLASS_MAPPINGS = {
    "AccumulateNode": AccumulateNode,
    "AccumulationHeadNode": AccumulationHeadNode,
    "AccumulationTailNode": AccumulationTailNode,
    "AccumulationToListNode": AccumulationToListNode,
    "ListToAccumulationNode": ListToAccumulationNode,
    "AccumulationGetLengthNode": AccumulationGetLengthNode,
    "AccumulationGetItemNode": AccumulationGetItemNode,
    "AccumulationSetItemNode": AccumulationSetItemNode,
    "DebugPrint": DebugPrint,
    "MakeListNode": MakeListNode,
    "GetFloatFromList": GetFloatFromList,
    "GetIntFromList": GetIntFromList
}

# Generate display names with configurable prefix
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "AccumulateNode": f"Accumulate {NODE_POSTFIX}",
    "AccumulationHeadNode": f"Accumulation Head {NODE_POSTFIX}",
    "AccumulationTailNode": f"Accumulation Tail {NODE_POSTFIX}",
    "AccumulationToListNode": f"Accumulation to List {NODE_POSTFIX}",
    "ListToAccumulationNode": f"List to Accumulation {NODE_POSTFIX}",
    "AccumulationGetLengthNode": f"Accumulation Get Length {NODE_POSTFIX}",
    "AccumulationGetItemNode": f"Accumulation Get Item {NODE_POSTFIX}",
    "AccumulationSetItemNode": f"Accumulation Set Item {NODE_POSTFIX}",
    "DebugPrint": f"Debug Print {NODE_POSTFIX}",
    "MakeListNode": f"Make List {NODE_POSTFIX}",
    "GetFloatFromList": f"Get Float From List {NODE_POSTFIX}",
    "GetIntFromList": f"Get Int From List {NODE_POSTFIX}",
}
