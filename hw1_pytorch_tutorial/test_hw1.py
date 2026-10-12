"""Meaningful reference, edge-case, gradient and integration checks for Homework 1."""

import ast
import json
import subprocess
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from notebook_tools import ROOT, load_definitions


torch.set_num_threads(2)
e1, e2, e3, e4 = [load_definitions(f"ex{i}") for i in range(1, 5)]


class TensorChecks(unittest.TestCase):
    def test_creation_shape_and_indexing(self):
        x = e1.make_tensor([[1, 2], [3, 4]], dtype=torch.float64)
        self.assertEqual(x.dtype, torch.float64)
        torch.testing.assert_close(e1.make_zeros((2, 2), dtype=x.dtype), torch.zeros_like(x))
        torch.testing.assert_close(e1.make_ones_like(x), torch.ones_like(x))
        torch.testing.assert_close(e1.make_arange(1, 8, 2), torch.tensor([1, 3, 5, 7]))
        torch.testing.assert_close(e1.make_linspace(0, 1, 3), torch.tensor([0., .5, 1.]))
        torch.testing.assert_close(e1.make_randn((2, 3), 7), e1.make_randn((2, 3), 7))
        self.assertEqual(e1.cast_dtype_and_move(x, torch.device("cpu"), torch.float32).dtype, torch.float32)
        self.assertEqual(e1.reshape_tensor(x, (4,)).shape, (4,))
        self.assertEqual(e1.view_tensor(x, (4,)).data_ptr(), x.data_ptr())
        with self.assertRaises(RuntimeError):
            e1.view_tensor(x.t(), (4,))
        self.assertTrue(e1.make_contiguous(x.t()).is_contiguous())
        self.assertEqual(e1.flatten_from_dim(x[None], 1).shape, (1, 4))
        self.assertEqual(e1.add_singleton_dim(x, 1).shape, (2, 1, 2))
        self.assertEqual(e1.remove_singleton_dims(x[None]).shape, x.shape)
        torch.testing.assert_close(e1.transpose_last_two(x), x.t())
        self.assertEqual(e1.permute_bhwc_to_bchw(torch.ones(2, 3, 4, 5)).shape, (2, 5, 3, 4))
        torch.testing.assert_close(e1.slice_rows(x, 1, 2), x[1:2])
        torch.testing.assert_close(e1.select_columns(x, [1, 0]), x[:, [1, 0]])
        torch.testing.assert_close(e1.get_diagonal(x), torch.tensor([1., 4.], dtype=x.dtype))
        changed = e1.set_subtensor(x, 0, 0, 9)
        self.assertEqual(x[0, 0].item(), 1)
        self.assertEqual(changed[0, 0].item(), 9)
        torch.testing.assert_close(e1.gather_rows(x, torch.tensor([1, 0])), x[[1, 0]])

    def test_reductions_vectorization(self):
        x = torch.arange(6.).reshape(2, 3)
        torch.testing.assert_close(e1.sum_over_dim(x, 0, True), x.sum(0, keepdim=True))
        torch.testing.assert_close(e1.mean_over_dim(x, 1), x.mean(1))
        values, indices = e1.max_over_dim(x, 1)
        torch.testing.assert_close(values, torch.tensor([2., 5.]))
        torch.testing.assert_close(indices, e1.argmax_over_dim(x, 1))
        torch.testing.assert_close(e1.broadcast_add_vector(x, torch.ones(3)), x + 1)
        self.assertEqual(e1.concat_tensors([x, x], 1).shape, (2, 6))
        self.assertEqual(e1.stack_tensors([x, x], 1).shape, (2, 2, 3))
        self.assertEqual(e1.repeat_tensor(x, (2, 1)).shape, (4, 3))
        expanded = e1.expand_tensor(x[:1], 3, -1)
        self.assertEqual(expanded.data_ptr(), x.data_ptr())
        torch.testing.assert_close(e1.cumsum_over_dim(x, 1), x.cumsum(1))
        torch.testing.assert_close(e1.where_select(x > 2, x, -x), torch.where(x > 2, x, -x))
        ids = torch.tensor([[0, 2], [1, 0]])
        self.assertEqual(e1.one_hot(ids, 3, torch.float64).shape, (2, 2, 3))
        self.assertEqual(e1.one_hot(ids, 3, torch.float64).dtype, torch.float64)
        torch.testing.assert_close(e1.scatter_add_1d(torch.tensor([1., 2., 3.]), torch.tensor([1, 1, 0]), 3), torch.tensor([3., 3., 0.]))
        torch.testing.assert_close(e1.batched_token_histogram(ids, 3), torch.tensor([[1, 0, 1], [1, 1, 0]]))
        mask = torch.tensor([[True], [False]])
        torch.testing.assert_close(e1.masked_mean(x, mask, 1), torch.tensor([1., 0.]))

    def test_einsum_and_causal_mask(self):
        x, w = torch.randn(2, 3, 4), torch.randn(4, 5)
        torch.testing.assert_close(e1.einsum_linear_btd_dh_to_bth(x, w), x @ w)
        torch.testing.assert_close(e1.einsum_pairwise_dot(x, x), x.square().sum(-1))
        q, k, v = [torch.randn(2, 3, 4, 5) for _ in range(3)]
        scores = e1.einsum_qk_scores(q, k)
        torch.testing.assert_close(scores, q @ k.transpose(-1, -2))
        torch.testing.assert_close(e1.einsum_apply_attention(scores, v), scores @ v)
        mask = e1.make_causal_mask(4)
        self.assertEqual(mask.sum().item(), 6)
        out = e1.apply_causal_mask(scores)
        self.assertTrue((out[..., mask] == -1e9).all())
        torch.testing.assert_close(out[..., ~mask], scores[..., ~mask])

    def test_stable_masked_softmax_and_gradients(self):
        x = torch.tensor([[10000., 10001., 9999.], [-5., 3., 1.]], requires_grad=True)
        torch.testing.assert_close(e1.stable_softmax(x), x.softmax(-1))
        mask = torch.tensor([[False, True, False], [True, True, True]])
        out = e1.masked_softmax(x, mask)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out.sum(-1), torch.tensor([1., 0.]))
        self.assertTrue((out[mask] == 0).all())
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[mask] == 0).all())
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: e1.masked_softmax(z, mask), (x.detach().double().requires_grad_(True),)))


class CoreChecks(unittest.TestCase):
    def test_autograd_does_not_accumulate_or_mutate_input(self):
        x = torch.randn(4, requires_grad=True)
        x.grad = torch.ones_like(x)
        for fn in (e2.grad_with_autograd_grad, e2.grad_with_backward):
            for _ in range(2):
                torch.testing.assert_close(fn(x), 2 * x)
            torch.testing.assert_close(x.grad, torch.ones_like(x))
        a, b = torch.randn(4), torch.randn(4)
        da, db = e2.grad_wrt_multiple_inputs(a, b)
        torch.testing.assert_close(da, 2 * a + b)
        torch.testing.assert_close(db, a)

    def test_datasets_padding_and_collation(self):
        tokens = torch.arange(20).reshape(2, 10)
        ds = e2.NextTokenDataset(tokens)
        inp, target = ds[1]
        torch.testing.assert_close(inp, tokens[1, :-1])
        torch.testing.assert_close(target, tokens[1, 1:])
        pair = e2.TensorPairDataset(tokens, tokens[:, 0])
        self.assertEqual(len(pair), 2)
        self.assertEqual(len(list(e2.make_dataloader(pair, 1, shuffle=False))), 2)
        crop1, crop2 = [e2.RandomCropSequenceDataset(tokens, 3, seed=9) for _ in range(2)]
        for _ in range(5):
            torch.testing.assert_close(crop1[0], crop2[0])
        padded = e2.pad_1d_sequences([torch.tensor([7., 8.]), torch.tensor([0.])], -1)
        torch.testing.assert_close(padded.tokens, torch.tensor([[7, 8], [0, -1]]))
        torch.testing.assert_close(padded.padding_mask, torch.tensor([[False, False], [False, True]]))
        self.assertEqual(e2.pad_1d_sequences([]).tokens.shape, (0, 0))
        self.assertEqual(e2.pad_1d_sequences([torch.empty(0), torch.empty(0)]).tokens.shape, (2, 0))
        batch = e2.collate_next_token_batch([(torch.tensor([0, 1]), torch.tensor([1, 2])),
                                            (torch.tensor([3]), torch.tensor([4]))])
        torch.testing.assert_close(batch['attention_mask'], ~batch['padding_mask'])
        self.assertTrue(batch['attention_mask'][0, 0])  # token 0 is not necessarily padding

    def test_adamw_matches_reference_over_multiple_steps(self):
        torch.manual_seed(123)
        params = [nn.Parameter(torch.randn(3, 4, dtype=torch.float64)), nn.Parameter(torch.tensor(2., dtype=torch.float64))]
        refs = [nn.Parameter(p.detach().clone()) for p in params]
        states = [e2.init_adamw_state(p) for p in params]
        optimizer = torch.optim.AdamW(refs, lr=.02, betas=(.8, .95), weight_decay=.1, eps=1e-7)
        for t in range(1, 6):
            grads = [torch.randn_like(p) for p in params]
            saved = [g.clone() for g in grads]
            for p, g in zip(refs, grads):
                p.grad = g.clone()
            optimizer.step()
            e2.adamw_step_many_(params, grads, states, .02, (.8, .95), 1e-7, .1)
            for p, ref, g, original, state in zip(params, refs, grads, saved, states):
                torch.testing.assert_close(p, ref, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(g, original)
                self.assertFalse(state.m.requires_grad)
                self.assertEqual(state.t, t)

    def test_initializers_and_training_step(self):
        self.assertEqual(e2.fan_in_fan_out(torch.empty(8, 3, 5, 5)), (75, 200))
        for initializer, reference in [(e2.xavier_uniform_, nn.init.xavier_uniform_), (e2.kaiming_uniform_, None)]:
            w = nn.Parameter(torch.empty(7, 4))
            torch.manual_seed(99)
            initializer(w)
            expected = torch.empty_like(w)
            torch.manual_seed(99)
            if reference:
                reference(expected)
            else:
                nn.init.kaiming_uniform_(expected, nonlinearity="relu")
            torch.testing.assert_close(w, expected)
        layer = e2.init_linear_(nn.Linear(2, 1), "zero")
        opt = torch.optim.SGD(layer.parameters(), lr=.1)
        batch = (torch.ones(4, 2), torch.ones(4, 1))
        initial = e2.train_step_mse(layer, batch, opt)
        self.assertLess(e2.train_step_mse(layer, batch, opt), initial)


class NetworkChecks(unittest.TestCase):
    def test_linear_embedding_and_parameter_gradients(self):
        layer = e3.Linear(4, 3).double()
        ref = nn.Linear(4, 3).double()
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(layer(x), ref(x))
        layer(x).sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(e3.Linear(4, 3, False)(x.float()).shape, (2, 5, 3))
        emb = e3.Embedding(5, 3)
        indices = torch.tensor([[1, 1, 4]])
        torch.testing.assert_close(emb(indices), emb.weight[indices])
        emb(indices).sum().backward()
        torch.testing.assert_close(emb.weight.grad[1], torch.full((3,), 2.))

    def test_normalization_reference_and_gradcheck(self):
        layer = e3.LayerNorm(4).double()
        ref = nn.LayerNorm(4).double()
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(layer(x), ref(x))
        self.assertTrue(torch.autograd.gradcheck(layer, (x,)))
        rms = e3.RMSNorm(4).double()
        torch.testing.assert_close(rms(x), nn.functional.rms_norm(x, (4,), rms.weight, rms.eps))
        self.assertTrue(torch.autograd.gradcheck(rms, (x,)))
        self.assertTrue(torch.isfinite(layer(torch.ones_like(x))).all())

    def test_dropout_modes_endpoints_and_expectation(self):
        x = torch.ones(100000)
        dropout = e3.Dropout(.25)
        out = dropout(x)
        self.assertLess(abs(out.mean().item() - 1), .015)
        self.assertLess(abs((out == 0).float().mean().item() - .25), .015)
        dropout.eval()
        self.assertIs(dropout(x), x)
        torch.testing.assert_close(e3.Dropout(0)(x), x)
        torch.testing.assert_close(e3.Dropout(1)(x), torch.zeros_like(x))
        with self.assertRaises(ValueError):
            e3.Dropout(1.1)

    def test_composition_and_cross_entropy(self):
        x = torch.randn(2, 3, 4)
        for depth in [1, 2, 3]:
            self.assertEqual(e3.MLP(4, 8, 5, depth, True)(x).shape, (2, 3, 5))
        ffn = e3.FeedForward(4)
        torch.testing.assert_close(e3.Residual(ffn)(x), x + ffn(x))
        logits = torch.tensor([[1000., 999., -1000.], [1001., 999., 1000.]], requires_grad=True)
        targets = torch.tensor([1, 2])
        loss = e3.cross_entropy_from_logits(logits, targets)
        torch.testing.assert_close(loss, nn.functional.cross_entropy(logits, targets))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_classifier_training_on_learnable_data(self):
        torch.manual_seed(3)
        x = torch.randn(128, 2)
        y = (x[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(x, y), batch_size=32)
        model = nn.Sequential(e3.MLP(2, 8, 8, 2), nn.GELU(), e3.ClassificationHead(8, 2))
        losses = e3.train_classifier(model, loader, loader, .04, 8)
        self.assertEqual(len(losses), 32)
        self.assertLess(model.training_history['test_losses'][-1], model.training_history['test_losses'][0])
        self.assertGreater(e3.evaluate_classifier(model, loader)[1], .9)


class ViTChecks(unittest.TestCase):
    def test_patch_pixel_order(self):
        image = torch.arange(32).reshape(1, 2, 4, 4)
        patches = e4.patchify(image, 2)
        expected = torch.stack([image[0, :, y:y + 2, x:x + 2].reshape(-1)
                                for y in (0, 2) for x in (0, 2)])[None]
        torch.testing.assert_close(patches, expected)
        with self.assertRaises(ValueError):
            e4.patchify(image, 3)

    def test_all_variants_forward_backward_and_parameter_budget(self):
        counts = []
        for variant in ['ffn', 'geglu', 'swiglu']:
            torch.manual_seed(0)
            model = e4.TinyViT(4, 16, 4, 2, 64, .1, variant)
            self.assertIsNot(model.blocks[0].mlp, model.blocks[1].mlp)
            logits = model(torch.rand(3, 1, 28, 28))
            self.assertEqual(logits.shape, (3, 10))
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            counts.append(sum(p.numel() for p in model.parameters()))
            model.eval()
            x = torch.rand(2, 1, 28, 28)
            torch.testing.assert_close(model(x), model(x))
        self.assertLess(max(counts) / min(counts), 1.03)
        self.assertEqual(counts[1], counts[2])

    def test_gating_formula(self):
        x = torch.randn(2, 3, 4)
        for variant, activation in [('geglu', nn.functional.gelu), ('swiglu', nn.functional.silu)]:
            layer = e4.GLUFeedForward(4, 6, 0., variant)
            expected = layer.projection(activation(layer.gate(x)) * layer.value(x))
            torch.testing.assert_close(layer(x), expected)

    def test_training_metrics(self):
        torch.manual_seed(0)
        x = torch.rand(8, 1, 28, 28)
        y = torch.arange(8) % 10
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        model = e4.TinyViT(7, 8, 2, 1, 16, 0., 'swiglu')
        cfg = e4.TrainConfig(epochs=2)
        result = e4.train_one_run('swiglu', model, loader, loader, cfg)
        self.assertEqual(len(result['train_losses']), 4)
        self.assertEqual(len(result['test_losses']), 2)
        self.assertEqual(result['final_test_accuracy'], result['test_accs'][-1])
        self.assertEqual(result['best_test_accuracy'], max(result['test_accs']))
        self.assertGreater(result['num_parameters'], 0)


class SubmissionChecks(unittest.TestCase):
    def test_no_stubs_and_original_signatures(self):
        def signatures(notebook):
            signatures = {}
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                tree = ast.parse(''.join(cell['source']))
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        signatures[node.name] = ast.dump(node.args)
                    elif isinstance(node, ast.ClassDef):
                        for method in node.body:
                            if isinstance(method, ast.FunctionDef):
                                signatures[f'{node.name}.{method.name}'] = ast.dump(method.args)
            return signatures
        for i in range(1, 5):
            relative = f'hw1_pytorch_tutorial/src/ex{i}.ipynb'
            original = json.loads(subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=ROOT.parent))
            current = json.loads((ROOT / 'src' / f'ex{i}.ipynb').read_text())
            for cell in current['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    self.assertNotIn('TODO', source)
                    self.assertNotIn('NotImplementedError', source)
            new_signatures = signatures(current)
            for name, signature in signatures(original).items():
                self.assertEqual(signature, new_signatures[name], f'ex{i}:{name}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
