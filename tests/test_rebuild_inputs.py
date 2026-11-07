"""Input isolation and cache-integrity tests, using authored synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from training.rebuild_inputs import select_inputs,certify_cache,REVISION,sha


class InputBoundaryTests(unittest.TestCase):
    def fixture(self):
        text='Authored test input, not a scientific observation.'
        record=dict(source_revision='revision-a',problem_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    model_id='target',reasoning_effort='default')
        row=dict(problem=text,model_id='target',reasoning_effort='default',is_robust=True)
        return {'records':[record]},row

    def run_select(self,manifest,rows):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return select_inputs(manifest,{'revision-a':p})

    def test_exact_selection_and_null(self):
        manifest,row=self.fixture();row['is_robust']=None
        self.assertIsNone(self.run_select(manifest,[row])[0]['label'])

    def test_missing_input_rejected(self):
        manifest,row=self.fixture();row['problem']='Different input'
        with self.assertRaises(ValueError):self.run_select(manifest,[row])

    def test_label_not_coerced(self):
        manifest,row=self.fixture();row['is_robust']=1
        with self.assertRaises(ValueError):self.run_select(manifest,[row])

    def test_conflicting_duplicate_rejected(self):
        manifest,row=self.fixture()
        with self.assertRaises(ValueError):self.run_select(manifest,[row,dict(row,is_robust=False)])

    def test_unselected_outcome_ignored(self):
        manifest,row=self.fixture()
        selected=self.run_select(manifest,[dict(row,problem='Unselected',is_robust={'invalid':'must not be inspected'}),row])
        self.assertEqual(len(selected),1)

    def test_cache_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'states.npy';np.save(p,np.ones((2,4096),dtype=np.float32))
            item=dict(states='states.npy',states_sha256=sha(p),problem_sha256=hashlib.sha256(b'text').hexdigest(),
                      lexical=[1.0]*4096,row={'offsets':[[0,1],[2,4]],'mask':[0,1]})
            index={'encoder_revision':REVISION,'layer':31,'records':{'pid':item}}
            self.assertEqual(certify_cache(root,index,'pid','text')[1].shape,(2,4096))
            index['layer']=30
            with self.assertRaises(ValueError):certify_cache(root,index,'pid','text')
            index['layer']=31
            with self.assertRaises(ValueError):certify_cache(root,index,'pid','other')
            p.write_bytes(b'tampered')
            with self.assertRaises(ValueError):certify_cache(root,index,'pid','text')


if __name__=='__main__':unittest.main()
