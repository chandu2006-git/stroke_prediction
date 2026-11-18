import urllib.request
import json
import urllib.error
from pathlib import Path

BASE = 'http://127.0.0.1:8000'
MODEL_RELATIVE = 'models/strokeguard_catboost.cbm'


def get(p):
    with urllib.request.urlopen(BASE + p) as r:
        return r.status, json.loads(r.read())


def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + '/api/predict', data=data, headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


passed = []

print('=== 1. HEALTH ===')
s, h = get('/api/health')
print('  status=%s | model_loaded=%s | model_path=%s | model_name=%s' % (
    s, h['model_loaded'], h['model_path'], h['model_name']))
assert s == 200
assert h['model_loaded'] is True
# model_path may use platform separators; compare as normalised strings
assert Path(h['model_path']).name == Path(MODEL_RELATIVE).name
passed.append('health')

print('=== 2. PREDICT ===')
payload = {
    'gender': 'Female', 'age': 55, 'hypertension': 1, 'heart_disease': 0,
    'ever_married': 'Yes', 'work_type': 'Private', 'residence_type': 'Urban',
    'avg_glucose_level': 110.0, 'bmi': 27.0, 'smoking_status': 'never smoked'
}
s, r = post(payload)
print('  status=%s | prediction=%s %s | prob=%s | model_name=%s' % (
    s, r['prediction'], r['prediction_label'], round(r['probability'], 4), r['model_name']))
assert s == 200
assert r['prediction'] in (0, 1)
passed.append('predict')

print('=== 3. SHAP ===')
shap = r['shap_explanation']
print('  available=%s | n_contributions=%s | base_value=%s' % (
    shap['available'], len(shap['contributions']), shap['base_value']))
assert shap['available'] is True
assert len(shap['contributions']) == 11
for c in shap['contributions']:
    print('    %-22s %+.4f (%s)' % (c['feature'], c['value'], c['direction']))
passed.append('shap')

print('=== 4. LIME ===')
lime = r['lime_explanation']
print('  available=%s | n_contributions=%s | intercept=%s' % (
    lime['available'], len(lime['contributions']), lime.get('intercept')))
assert lime['available'] is True
assert len(lime['contributions']) == 11
for c in lime['contributions']:
    print('    %-40s %+.4f (%s)' % (c['feature'], c['weight'], c['direction']))
passed.append('lime')

print('=== 5. MISSING BMI ===')
payload2 = dict(payload)
payload2['bmi'] = None
s2, r2 = post(payload2)
print('  status=%s | bmi_missing=%s' % (s2, r2['patient_summary']['bmi_missing']))
assert s2 == 200
assert r2['patient_summary']['bmi_missing'] == 1
passed.append('bmi_missing')

print('=== ALL VERIFICATION CHECKS PASSED ===')
print('PASSED:', ', '.join(passed))
