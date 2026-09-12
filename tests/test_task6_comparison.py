import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.task6_comparison import split_metrics, roc_summary, donor_frequencies, validate_tables

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.pred=pd.DataFrame({'split_id':[0,0,1,1], 'donor_id':['a','b','c','d'],
            'y_true':[0,1,0,1], 'score':[.1,.2,.8,.9]})
        self.freq=self.pred[['split_id','donor_id','y_true']].assign(
            frequency=[0.,1.,0.,1.], n_cells=10, n_selected_cells=[0,10,0,10])

    def test_roc_is_averaged_by_split_not_pooled(self):
        metrics=split_metrics(self.pred,self.freq)
        self.assertTrue(metrics.network_auc.eq(1).all())
        self.assertEqual(roc_auc_score(self.pred.y_true,self.pred.score),.75)
        curve,auc,count=roc_summary(self.pred,'score')
        self.assertEqual(auc,1);self.assertEqual(count,2)
        np.testing.assert_array_equal(curve,np.ones_like(curve))

    def test_frequency_direction_missingness_and_constants(self):
        reverse=self.freq.assign(frequency=1-self.freq.frequency,
                                 n_selected_cells=10-self.freq.n_selected_cells)
        metrics=split_metrics(self.pred,reverse)
        self.assertTrue(metrics.frequency_auc.eq(0).all())
        self.assertTrue(metrics.frequency_effect.eq(-1).all())
        missing=reverse.copy();missing.loc[missing.split_id.eq(1),['frequency','n_selected_cells']]=np.nan
        m=split_metrics(self.pred,missing)
        self.assertTrue(np.isnan(m.loc[m.split_id.eq(1),'frequency_auc'].item()))
        self.assertEqual(roc_summary(missing,'frequency')[2],1)
        constant=self.freq.assign(frequency=0.,n_selected_cells=0)
        self.assertTrue(split_metrics(self.pred,constant).frequency_auc.eq(.5).all())

    def test_one_point_per_donor_with_unequal_test_visits(self):
        f=pd.DataFrame({'donor_id':['a','a','a','b'],'y_true':[0,0,0,1],
                        'frequency':[0.,0.,0.,1.]})
        donors=donor_frequencies(f)
        self.assertEqual(len(donors),2)
        self.assertEqual(donors.frequency.mean(),.5)
        self.assertEqual(donors.valid_test_visits.tolist(),[3,1])

    def test_invalid_pairing_and_numerators_rejected(self):
        with self.assertRaises(ValueError):
            validate_tables(pd.concat([self.pred,self.pred.iloc[[0]]]),self.freq)
        wrong=self.freq.copy();wrong.loc[0,'y_true']=1
        with self.assertRaises(ValueError):validate_tables(self.pred,wrong)
        with self.assertRaises(AssertionError):
            validate_tables(self.pred,self.freq.assign(n_selected_cells=5))

    def test_inner_validation_donor_rejected_from_threshold_reference(self):
        splits=pd.DataFrame({'split_id':[0]*4,'split_seed':[42]*4,
            'donor_id':['train','validation','a','b'],'label':[0,1,0,1],
            'outer_partition':['train','train','test','test'],'inner_fold':[1,0,-1,-1]})
        pred=self.pred.loc[self.pred.split_id.eq(0)].assign(split_seed=42,
            decision_threshold=.5,y_pred=0,selected_inner_fold=0)
        freq=self.freq.loc[self.freq.split_id.eq(0)].assign(training_donors='train')
        validate_tables(pred,freq,splits)
        with self.assertRaises(ValueError):
            validate_tables(pred,freq.assign(training_donors='train;validation'),splits)

if __name__=='__main__':unittest.main()
