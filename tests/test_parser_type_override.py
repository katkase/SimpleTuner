import unittest

from simpletuner.helpers.configuration import cmd_args


class TestParserTypeOverride(unittest.TestCase):
    def setUp(self):
        cmd_args._ARG_PARSER_CACHE = None

    def test_optimizer_beta1_uses_float_type(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--optimizer_beta1" in action.option_strings)
        self.assertIs(action.type, float)

    def test_optimizer_beta2_uses_float_type(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--optimizer_beta2" in action.option_strings)
        self.assertIs(action.type, float)

    def test_cosine_decay_peak_float_fields_parse_scientific_notation(self):
        parser = cmd_args.get_argument_parser()
        eta_action = next(action for action in parser._actions if "--eta_min" in action.option_strings)
        warmup_start_action = next(action for action in parser._actions if "--lr_warmup_start" in action.option_strings)
        floor_action = next(action for action in parser._actions if "--lr_floor" in action.option_strings)
        floor_start_action = next(action for action in parser._actions if "--lr_floor_start" in action.option_strings)
        self.assertIs(eta_action.type, float)
        self.assertIs(warmup_start_action.type, float)
        self.assertIs(floor_action.type, float)
        self.assertIs(floor_start_action.type, float)
        self.assertAlmostEqual(eta_action.type("4e-7"), 4e-7)
        self.assertAlmostEqual(warmup_start_action.type("4e-7"), 4e-7)
        self.assertAlmostEqual(floor_action.type("4e-7"), 4e-7)
        self.assertAlmostEqual(floor_start_action.type("4e-7"), 4e-7)
