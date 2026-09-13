import copy
import unittest
import torch
from soku_iql.bc_core.config import MODEL_DEFAULTS
from soku_iql.bc_core.models import BCNetwork
from soku_iql.bc_core.checkpoint import adapt_latest_bc


class LatestBCAdapterTests(unittest.TestCase):
    def source(self):
        model = BCNetwork({**MODEL_DEFAULTS, 'temporal_mode': 'tcn256'})
        spec = copy.deepcopy(model.spec)
        spec['network_version'] = 'soku_bc_tcn256_joint144_v1'
        return dict(algorithm='bc', network_version=spec['network_version'], spec=spec,
                    config={'model': copy.deepcopy(spec['model'])}, model=model.state_dict())

    def test_alias_preserves_weights_and_source(self):
        source = self.source()
        adapted = adapt_latest_bc(source)
        self.assertEqual(source['network_version'], 'soku_bc_tcn256_joint144_v1')
        self.assertEqual(adapted['network_version'], 'soku_iql_tcn256_joint144_v1')
        self.assertIs(source['model'], adapted['model'])

    def test_reject_semantic_change(self):
        source = self.source()
        source['spec']['temporal']['receptive_field_frames'] = 32
        with self.assertRaises(ValueError):
            adapt_latest_bc(source)

    def test_reject_missing_weight(self):
        source = self.source()
        source['model'].pop(next(iter(source['model'])))
        with self.assertRaises(RuntimeError):
            adapt_latest_bc(source)
