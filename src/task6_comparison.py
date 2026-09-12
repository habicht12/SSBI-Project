"""Gemeinsame Donor-Metriken und Abbildungen für die vier Aufgabe-6-Modelle."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, balanced_accuracy_score

MODELS = {
    'cellcnn': ('CellCNN · Top-1 %', '#087F8C'),
    'quadratic_top1': ('Quadratic · Top-1 %', '#4956A3'),
    'quadratic_soft': ('Quadratic · Soft-α', '#DD8D29'),
    'prototype_soft': ('Prototyp/ReLU · Soft-α', '#AF4D35'),
}
SHORT_LABELS = ['CellCNN', 'Quadratic\nTop-1 %', 'Quadratic\nSoft-α', 'Prototyp\nSoft-α']
PREFIXES = {'cellcnn': 'task6_baseline_', 'quadratic_top1': 'task6_quadratic_',
            'quadratic_soft': 'task6_quadratic_learnable_', 'prototype_soft': 'task6_learnable_pooling_relu_'}
GRID = np.linspace(0, 1, 501)

def validate_tables(predictions, frequencies, donor_splits=None):
    keys = ['split_id', 'donor_id']
    for table in [predictions, frequencies]:
        if table.empty or table.duplicated(keys).any():
            raise ValueError('Leere Tabelle oder doppelte Testauftritte.')
    merged = predictions[keys + ['y_true']].merge(frequencies[keys + ['y_true']], on=keys,
                                               suffixes=('_prediction', '_frequency'), validate='one_to_one', how='outer')
    if len(merged) != len(predictions) or len(merged) != len(frequencies) or not merged.y_true_prediction.eq(merged.y_true_frequency).all():
        raise ValueError('Vorhersagen und Frequenzen gehören nicht zu denselben Spendern/Labels.')
    if not predictions.score.between(0, 1).all() or not frequencies.n_cells.between(1, 20000).all():
        raise ValueError('Ungültige Scores oder Zellzahlen.')
    available = frequencies.frequency.notna()
    if not frequencies.loc[available, 'frequency'].between(0, 1).all():
        raise ValueError('Frequenz außerhalb [0,1].')
    if 'n_selected_cells' in frequencies:
        selected = frequencies.loc[available, 'n_selected_cells']
        if selected.isna().any() or not selected.eq(selected.astype(int)).all():
            raise ValueError('Ungültige Anzahl ausgewählter Zellen.')
        np.testing.assert_allclose(frequencies.loc[available, 'frequency'],
            selected / frequencies.loc[available, 'n_cells'], rtol=0, atol=1e-14)
    if donor_splits is not None:
        from src.task4_artifacts import validate_prediction_splits
        validate_prediction_splits(predictions, donor_splits)
        for sid, group in predictions.groupby('split_id'):
            folds = group.selected_inner_fold.unique()
            if len(folds) != 1:
                raise ValueError('Mehrere ausgewählte Inner-Folds im selben Split.')
            split = donor_splits.loc[donor_splits.split_id.eq(sid)]
            train_ids = set(split.loc[split.outer_partition.eq('train') & split.inner_fold.ne(folds[0]), 'donor_id'])
            for donors in frequencies.loc[frequencies.split_id.eq(sid), 'training_donors']:
                if set(donors.split(';')) != train_ids:
                    raise ValueError('Half-Max-Herkunft enthält falsche Trainingsspender.')

def split_metrics(predictions, frequencies, donor_splits=None):
    """Jeder äußere Split ist eine Auswertung; niemals alle Testauftritte poolen."""
    validate_tables(predictions, frequencies, donor_splits)
    rows = []
    for sid, pred in predictions.groupby('split_id', sort=True):
        freq = frequencies.loc[frequencies.split_id.eq(sid)]
        valid = freq.frequency.notna().all()
        rows.append(dict(split_id=sid, network_auc=roc_auc_score(pred.y_true, pred.score),
            average_precision=average_precision_score(pred.y_true, pred.score),
            balanced_accuracy=balanced_accuracy_score(pred.y_true, pred.score.ge(.5).astype(int)),
            frequency_auc=roc_auc_score(freq.y_true, freq.frequency) if valid else np.nan,
            frequency_effect=(freq.loc[freq.y_true.eq(1), 'frequency'].mean() - freq.loc[freq.y_true.eq(0), 'frequency'].mean()) if valid else np.nan,
            phenotype_available=bool(valid)))
    return pd.DataFrame(rows)

def roc_summary(table, score, split_ids=None):
    curves, aucs = [], []
    for sid, group in table.groupby('split_id', sort=True):
        if split_ids is not None and sid not in split_ids:
            continue
        if group[score].isna().any():
            continue
        fpr, tpr, _ = roc_curve(group.y_true, group[score], drop_intermediate=False)
        curves.append(np.interp(GRID, fpr, tpr))
        aucs.append(roc_auc_score(group.y_true, group[score]))
    if not curves:
        return np.full_like(GRID, np.nan), np.nan, 0
    return np.mean(curves, axis=0), float(np.mean(aucs)), len(curves)

def donor_frequencies(frequencies):
    """Zuerst Testauftritte je Spender mitteln: ein Punkt je biologischem Spender."""
    if frequencies.groupby('donor_id').y_true.nunique().gt(1).any():
        raise ValueError('Widersprüchliche Donorlabels.')
    return frequencies.groupby(['donor_id', 'y_true'], as_index=False).agg(
        frequency=('frequency', 'mean'), valid_test_visits=('frequency', 'count'))

def _style():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#172B43', 'text.color': '#172B43', 'axes.titleweight': 'bold',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

def _save(fig, output, name):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(output / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return output / f'{name}.png'

def _roc(ax, table, score, color, label, split_ids=None, linestyle='-'):
    mean, auc, count = roc_summary(table, score, split_ids)
    if count:
        # Doppelte FPR=0 bewahrt den vertikalen Sprung einer perfekten ROC.
        ax.plot(np.r_[0, GRID], np.r_[0, mean], color=color, lw=2, linestyle=linestyle,
                label=f'{label}: mittlere AUC {auc:.3f} (n={count})')
    ax.plot([0, 1], [0, 1], '--', color='#a6afb9', lw=.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Falsch-positiv-Rate', ylabel='Richtig-positiv-Rate')

def _boxes(ax, values, labels, colors, *, zero=False, limits=None):
    arrays = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in values]
    positions = np.arange(len(arrays))
    boxes = ax.boxplot(arrays, positions=positions, widths=.48, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(610)
    for patch, array, pos, color in zip(boxes['boxes'], arrays, positions, colors):
        patch.set(facecolor=color, alpha=.22, edgecolor=color)
        ax.scatter(pos + rng.uniform(-.12, .12, len(array)), array, s=12, color=color, alpha=.5, linewidths=0)
    ax.set_xticks(positions, labels)
    if zero:
        ax.axhline(0, color='#526579', linestyle='--', lw=.8)
    if limits:
        ax.set_ylim(*limits)
    ax.grid(axis='y', alpha=.15)

def _donors(ax, frequencies, color):
    donors = donor_frequencies(frequencies)
    rng = np.random.default_rng(611)
    for label, marker in [(0, 'o'), (1, '^')]:
        group = donors.loc[donors.y_true.eq(label) & donors.frequency.notna()]
        ax.scatter(label + rng.uniform(-.13, .13, len(group)), 100 * group.frequency, color=color, marker=marker, s=35)
        if len(group):
            ax.plot([label-.18, label+.18], [100*group.frequency.mean()]*2, color='#172B43', lw=2)
    ax.set_xticks([0, 1], ['CMV−', 'CMV+'])
    ax.set_ylabel('Mittlere Testfrequenz je Spender (%)')
    ax.grid(axis='y', alpha=.15)
    return donors

def plot_model_results(predictions, frequencies, output, model_key, donor_splits=None):
    _style(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    label, color = MODELS[model_key]
    metrics = split_metrics(predictions, frequencies, donor_splits)
    metrics.to_csv(output/'split_metrics.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    _roc(axes[0,0], predictions, 'score', color, 'Netzwerk'); axes[0,0].legend(fontsize=8, loc='lower right'); axes[0,0].set_title('Netzwerk-ROC')
    _roc(axes[0,1], frequencies, 'frequency', color, 'Frequency'); axes[0,1].legend(fontsize=8, loc='lower right'); axes[0,1].set_title('Half-Max-Frequency-ROC')
    _boxes(axes[0,2], [metrics[c] for c in ['network_auc','average_precision','balanced_accuracy']], ['ROC-AUC','AP','BA'], [color]*3, limits=(-.03,1.03)); axes[0,2].set_title('Klassifikation je Split')
    donors = _donors(axes[1,0], frequencies, color); axes[1,0].set_title('Ein Punkt je Spender')
    donors.to_csv(output/'donor_frequencies.csv', index=False)
    _boxes(axes[1,1], [metrics.frequency_auc], ['Frequency-AUC'], [color], limits=(-.03,1.03)); axes[1,1].set_title('Populationsassoziation je Split')
    _boxes(axes[1,2], [100*metrics.frequency_effect], ['CMV+ minus CMV−'], [color], zero=True); axes[1,2].set_ylabel('Prozentpunkte'); axes[1,2].set_title('Frequency-Effect je Split')
    fig.suptitle(f'{label} · {len(metrics)} äußere Splits · gated_alive', fontsize=16)
    fig.supxlabel('Boxen: Q1–Q3, Median; Punkte: Splits. Donorfrequenzen mitteln Testauftritte. Keine unabhängigen Wiederholungen.', fontsize=9)
    return _save(fig, output, 'dashboard')

def compare_all(tables, output):
    """Vier Modelle auf denselben Splits; Frequency-Vergleiche nur gemeinsam bestimmbar."""
    _style(); plt.rcParams.update({'font.size': 14})
    tables=Path(tables); output=Path(output); output.mkdir(parents=True, exist_ok=True)
    donor_splits=pd.read_csv(tables/'task4_donor_splits.csv')
    data={}; metric_parts=[]
    for key in MODELS:
        prefix=PREFIXES[key]
        pred_path=tables/(prefix+'predictions.csv')
        if key=='cellcnn': pred_path=tables/'task4_cellcnn_predictions_gated_alive_full.csv'
        p=pd.read_csv(pred_path, float_precision='round_trip')
        f=pd.read_csv(tables/(prefix+'frequencies.csv'), float_precision='round_trip')
        p=p.loc[p.split_id.isin(f.split_id)].copy()
        m=split_metrics(p,f,donor_splits)
        if set(m.split_id)!=set(range(100)):
            raise ValueError(f'{key}: erwartet werden alle Splits 0–99.')
        data[key]=(p,f,m); metric_parts.append(m.assign(model=key))
    metrics=pd.concat(metric_parts,ignore_index=True)
    common=set.intersection(*(set(m.loc[m.phenotype_available,'split_id']) for _,_,m in data.values()))
    if not common: raise ValueError('Keine gemeinsam bestimmbaren Frequency-Splits.')
    metrics['common_frequency_split']=metrics.split_id.isin(common)
    metrics.to_csv(output/'split_metrics.csv',index=False)
    pd.concat([p.assign(model=k) for k,(p,f,m) in data.items()]).to_csv(output/'predictions.csv',index=False)
    pd.concat([f.rename(columns={'response_threshold':'halfmax_threshold'}).assign(model=k)
               for k,(p,f,m) in data.items()]).to_csv(output/'frequencies.csv',index=False)
    summary=[]; rocs=[]
    for key,(p,f,m) in data.items():
        row={'model':key,'label':MODELS[key][0],'network_splits':len(m),
             'frequency_splits_available':int(m.phenotype_available.sum()),'frequency_splits_common':len(common)}
        for column in ['network_auc','average_precision','balanced_accuracy','frequency_auc','frequency_effect']:
            values=m.loc[m.split_id.isin(common),column] if column.startswith('frequency') else m[column]
            for statistic,value in [('mean',values.mean()),('median',values.median()),('q25',values.quantile(.25)),('q75',values.quantile(.75))]:row[f'{column}_{statistic}']=value
        summary.append(row)
        for kind,table,score,ids in [('network',p,'score',None),('frequency',f,'frequency',common)]:
            mean,auc,count=roc_summary(table,score,ids)
            rocs.extend({'model':key,'kind':kind,'fpr':x,'mean_tpr':y,'mean_split_auc':auc,'n_splits':count} for x,y in zip(GRID,mean))
    summary=pd.DataFrame(summary);summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(rocs).to_csv(output/'mean_rocs.csv',index=False)
    colors=[v[1] for v in MODELS.values()]
    for kind,score,title in [('network','score','Netzwerk-ROC · Mittel über 100 Splits'),('frequency','frequency',f'Half-Max-Frequency-ROC · {len(common)} gemeinsame Splits')]:
        fig,ax=plt.subplots(figsize=(7.2,3.9),layout='constrained')
        for key,(p,f,m) in data.items():_roc(ax,p if kind=='network' else f,score,MODELS[key][1],MODELS[key][0],None if kind=='network' else common,linestyle='--' if key=='quadratic_soft' else '-')
        ax.set_title(title);ax.legend(fontsize=11,loc='lower right');_save(fig,output,f'{kind}_roc')
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,column,title in zip(axes,['network_auc','average_precision','balanced_accuracy'],['ROC-AUC','Average Precision','Balanced Accuracy (Schwelle 0,5)']):
        _boxes(ax,[data[k][2][column] for k in MODELS],SHORT_LABELS,colors,limits=(-.03,1.03));ax.set_title(title)
    fig.supxlabel('100 überlappende Splits; Boxen Q1–Q3, Linie Median, Punkte Splits. Keine Konfidenzintervalle.',fontsize=9)
    _save(fig,output,'network_metrics')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,column,title,factor in [(axes[0],'frequency_auc','Frequency-AUC',1),(axes[1],'frequency_effect','Frequency-Effect (Prozentpunkte)',100)]:
        _boxes(ax,[factor*data[k][2].loc[data[k][2].split_id.isin(common),column] for k in MODELS],SHORT_LABELS,colors,zero=column=='frequency_effect',limits=(-.03,1.03) if column=='frequency_auc' else None);ax.set_title(title)
    fig.supxlabel(f'{len(common)} gemeinsam bestimmbare Splits; sonstige Ausfälle bleiben NaN.',fontsize=9);_save(fig,output,'frequency_metrics')
    base=data['cellcnn'][2].set_index('split_id');differences=[]
    for key in list(MODELS)[1:]:
        m=data[key][2].set_index('split_id')
        for sid in m.index:
            differences.append({'model':key,'split_id':sid,'network_auc_delta':m.loc[sid,'network_auc']-base.loc[sid,'network_auc'],
                'frequency_auc_delta':m.loc[sid,'frequency_auc']-base.loc[sid,'frequency_auc'] if sid in common else np.nan})
    delta=pd.DataFrame(differences);delta.to_csv(output/'paired_differences.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.5),layout='constrained')
    for ax,column,title in zip(axes,['network_auc_delta','frequency_auc_delta'],['Netzwerk-AUC minus CellCNN','Frequency-AUC minus CellCNN']):
        _boxes(ax,[delta.loc[delta.model.eq(k),column] for k in list(MODELS)[1:]],SHORT_LABELS[1:],colors[1:],zero=True);ax.set_title(title)
    _save(fig,output,'paired_differences')
    fig,axes=plt.subplots(1,4,figsize=(14,4.2),layout='constrained',sharey=True)
    donor_parts=[]
    for ax,(key,(p,f,m)) in zip(axes,data.items()):
        donors=_donors(ax,f.loc[f.split_id.isin(common)],MODELS[key][1]);ax.set_title(MODELS[key][0],fontsize=11);donor_parts.append(donors.assign(model=key))
    for ax in axes[1:]:ax.set_ylabel('')
    fig.supxlabel('Ein Punkt je Spender: Mittel über verfügbare Testauftritte in gemeinsamen Splits. Die Population kann je Split variieren.',fontsize=9)
    _save(fig,output,'donor_frequencies');pd.concat(donor_parts).to_csv(output/'donor_frequencies.csv',index=False)
    alpha_frames=[]
    for key in ['quadratic_soft','prototype_soft']:
        a=pd.read_csv(tables/(PREFIXES[key]+'alphas.csv'));alpha_frames.append(a.assign(model=key))
    alphas=pd.concat(alpha_frames,ignore_index=True);alphas.to_csv(output/'learned_alphas.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.3),layout='constrained')
    _boxes(ax,[100*alphas.loc[alphas.model.eq(k),'alpha'] for k in ['quadratic_soft','prototype_soft']],['Quadratic Soft-α','Prototyp Soft-α'],[MODELS[k][1] for k in ['quadratic_soft','prototype_soft']]);ax.axhline(1,color='#526579',ls='--',lw=1,label='Initialisierung: 1 %');ax.set(ylabel='Gelernter Poolinganteil α (%)',title='Alpha aller ausgewählten Modellfilter');ax.legend()
    _save(fig,output,'learned_alphas')
    provenance={'split_ids':list(range(100)),'common_frequency_splits':sorted(map(int,common)),
        'independent_donors':int(donor_splits.donor_id.nunique()),'frequency_rule':'strict_response_above_half_inner_training_maximum',
        'roc_aggregation':'interpolate each split ROC, then average; legend reports mean original split AUC',
        'donor_frequency_aggregation':'one point per donor, averaging their test visits within common frequency splits',
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.glob('*.csv')},
        'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return summary

def export_slide_assets(tables, output):
    """Folienwerte aus den geprüften Vergleichstabellen übernehmen; kein Training."""
    import shutil
    tables=Path(tables); comparison=tables/'comparison'; output=Path(output)
    figures=output/'figures'; data=output/'data'
    figures.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(comparison/'summary.csv').set_index('model')
    delta=pd.read_csv(comparison/'paired_differences.csv')
    metrics=pd.read_csv(comparison/'split_metrics.csv')
    alphas=pd.read_csv(comparison/'learned_alphas.csv')
    names={'cellcnn':'CellCNN','quadratic_top1':r'Quadratic Top-1\,\%',
           'quadratic_soft':r'Quadratic Soft-$\alpha$','prototype_soft':r'Prototyp Soft-$\alpha$'}
    short={'cellcnn':'CellCNN','quadratic_top1':r'Quad.~Top-1\,\%',
           'quadratic_soft':r'Quad.~Soft-$\alpha$','prototype_soft':'Prototyp'}
    def fmt(value,digits=4,signed=False):
        return format(float(value),('+' if signed else '')+f'.{digits}f').replace('.',',')
    best_network=summary.network_auc_mean.idxmax()
    network_winners=summary.index[np.isclose(summary.network_auc_mean,summary.network_auc_mean.max(),rtol=0,atol=1e-12)]
    best_ap=summary.average_precision_mean.idxmax();best_ba=summary.balanced_accuracy_mean.idxmax()
    best_frequency=summary.frequency_auc_mean.idxmax();best_effect=summary.frequency_effect_mean.idxmax()
    n_common=int(summary.frequency_splits_common.iloc[0])
    qdelta=delta.loc[delta.model.eq('quadratic_soft'),'network_auc_delta']
    wide=metrics.pivot(index='split_id',columns='model',values='network_auc')
    soft_minus_hard=(wide.quadratic_soft-wide.quadratic_top1).mean()
    alpha_medians=alphas.groupby('model').alpha.median()*100
    macros={
        'CommonFrequencySplits':str(n_common),
        'NetworkTakeaway':f'Höchste mittlere Network-AUC: {" und ".join(names[k] for k in network_winners)} ({fmt(summary.loc[best_network,"network_auc_mean"])}).',
        'ClassificationTakeaway':f'Höchste Mittelwerte: AP bei {names[best_ap]} ({fmt(summary.loc[best_ap,"average_precision_mean"],3)}); BA bei {names[best_ba]} ({fmt(summary.loc[best_ba,"balanced_accuracy_mean"],3)}).',
        'FrequencyTakeaway':f'Höchste mittlere Frequency-AUC: {names[best_frequency]} ({fmt(summary.loc[best_frequency,"frequency_auc_mean"])}).',
        'EffectTakeaway':f'Größter mittlerer Frequency-Effect: {names[best_effect]} ({fmt(100*summary.loc[best_effect,"frequency_effect_mean"],3)} Prozentpunkte).',
        'PairedTakeaway':f'Quadratic Soft-$\\alpha$ gegen CellCNN: {int((qdelta>0).sum())} besser, {int((qdelta==0).sum())} gleich, {int((qdelta<0).sum())} schlechter; mittlere Differenz {fmt(qdelta.mean(),signed=True)}.',
        'AlphaTakeaway':f'Median $\\alpha$: Quadratic {fmt(alpha_medians.quadratic_soft,2)}\\,\\%; Prototyp {fmt(alpha_medians.prototype_soft,2)}\\,\\%. Alpha beschreibt das Pooling, nicht die gemessene Zellhäufigkeit.',
        'FinalTakeaway':f'Quadratic Soft minus Hard: mittlere Network-AUC-Differenz {fmt(soft_minus_hard,signed=True)}. Eine unabhängige Bestätigung benötigt neue Spender.',
    }
    if abs(soft_minus_hard)<1e-12:
        macros['FinalTakeaway']=f'Quadratic Hard und Soft erreichen beide {fmt(summary.loc["quadratic_top1","network_auc_mean"])} mittlere Network-AUC. Lernbares Soft-Pooling erhöht sie hier nicht.'
    (data/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n')
    for kind in ['network','frequency']:
        lines=[r'\begin{tabular}{@{}lrr@{}}\toprule',
               r'Modell & Mittel & Median\\\midrule' if kind=='network' else r'Modell & AUC & Effekt*\\\midrule']
        for key in MODELS:
            if kind=='network':values=[summary.loc[key,'network_auc_mean'],summary.loc[key,'network_auc_median']]
            else:values=[summary.loc[key,'frequency_auc_mean'],100*summary.loc[key,'frequency_effect_mean']]
            lines.append(short[key]+' & '+' & '.join(fmt(v,4 if kind=='network' or j==0 else 3) for j,v in enumerate(values))+r'\\')
        lines += [r'\bottomrule\end{tabular}']
        if kind=='frequency':lines += [r'\par\smallskip{\tiny *Prozentpunkte; gemeinsame Splits.}']
        (data/f'{kind}_table.tex').write_text('\n'.join(lines)+'\n')
    for name in ['network_roc','frequency_roc','network_metrics','frequency_metrics','paired_differences','donor_frequencies','learned_alphas']:
        shutil.copy2(comparison/f'{name}.pdf',figures/f'{name}.pdf')
    for name in ['summary.csv','paired_differences.csv']:
        shutil.copy2(comparison/name,data/name)
    provenance={'source':str(comparison),'common_frequency_splits':n_common,
        'input_sha256':{name:hashlib.sha256((comparison/name).read_bytes()).hexdigest()
                       for name in ['summary.csv','paired_differences.csv','split_metrics.csv','learned_alphas.csv']},
        'takeaways':macros,'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (data/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    return macros

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    tables=root/'results/tables/task6_comparison_100'
    print(compare_all(tables,tables/'comparison').to_string(index=False))
    export_slide_assets(tables,root/'praesentation/aufgabe_06_vergleich')
