(function () {
  // ------------------------------------------------------------------ Config
  const API_BASE = (window.API_BASE_URL && window.API_BASE_URL.replace(/\/$/, '')) ||
    'http://127.0.0.1:8000';

  // ------------------------------------------------------------------ Elements
  const form = document.getElementById('assessment-form');
  const submitBtn = document.getElementById('btn-submit');
  const resetBtn = document.getElementById('btn-reset');
  const newAssessmentBtn = document.getElementById('btn-new-assessment');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results');

  const predictionBadge = document.getElementById('prediction-badge');
  const predictionLabel = document.getElementById('prediction-label');
  const probabilityValue = document.getElementById('probability-value');
  const modelNameEl = document.getElementById('model-name');
  const timestampEl = document.getElementById('prediction-timestamp');
  const explanationStatusEl = document.getElementById('explanation-status');
  const shapList = document.getElementById('shap-list');
  const limeList = document.getElementById('lime-list');
  const patientSummary = document.getElementById('patient-summary');

  // ------------------------------------------------------------------ State
  let submitting = false;

  // ------------------------------------------------------------------ Helpers
  function showLoading(show) {
    submitting = show;
    if (show) {
      loadingOverlay.classList.add('open');
      submitBtn.disabled = true;
      resultsSection.style.display = 'none';
    } else {
      loadingOverlay.classList.remove('open');
      submitBtn.disabled = false;
    }
  }

  function clearErrors() {
    form.querySelectorAll('.field').forEach(function (field) {
      field.classList.remove('invalid');
    });
  }

  function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
  }

  function formatLabel(name) {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
  }

  // ------------------------------------------------------------------ Renderers
  function renderPrediction(result) {
    const isStroke = result.prediction === 1;
    const prob = result.probability;

    predictionBadge.textContent = result.prediction_label;
    predictionBadge.className = 'prediction__badge ' +
      (isStroke ? 'prediction__badge--stroke' : 'prediction__badge--nostroke');
    predictionLabel.textContent = result.prediction_label;
    predictionLabel.style.color = isStroke ? '#7a2d33' : '#174C3C';

    probabilityValue.textContent = (prob * 100).toFixed(1) + '%';
    probabilityValue.style.color = isStroke ? '#7a2d33' : '#174C3C';

    modelNameEl.textContent = result.model_name || 'CatBoost Classifier';
    timestampEl.textContent = result.timestamp ? new Date(result.timestamp).toLocaleString() : '—';

    const explCount = [
      result.shap_explanation && result.shap_explanation.available,
      result.lime_explanation && result.lime_explanation.available,
    ].filter(Boolean).length;
    explanationStatusEl.textContent = explCount > 0 ? explCount + ' available' : 'None';
    explanationStatusEl.style.color = explCount > 0 ? '#174C3C' : '#8a8678';
  }

  function renderContributions(listEl, contributions, valueKey) {
    listEl.innerHTML = '';
    if (!contributions || contributions.length === 0) {
      listEl.innerHTML = '<div class="explainer__unavailable"><p>No contributions available.</p></div>';
      return;
    }

    contributions.forEach(function (item) {
      const dir = item.direction || 'neutral';
      const val = item[valueKey] != null ? item[valueKey] : 0;
      const absVal = Math.abs(val);
      const barWidth = Math.min(absVal * 100, 100);

      const row = document.createElement('div');
      row.className = 'contrib__row';

      const label = document.createElement('div');
      label.className = 'contrib__label';

      const name = document.createElement('span');
      name.className = 'contrib__name';
      name.textContent = escapeHtml(item.feature);

      const valSpan = document.createElement('span');
      valSpan.className = 'contrib__val contrib__val--' + dir;
      valSpan.textContent = val >= 0 ? '+' + val.toFixed(4) : val.toFixed(4);

      label.appendChild(name);
      label.appendChild(valSpan);

      const bar = document.createElement('div');
      bar.className = 'contrib__bar';

      const fill = document.createElement('div');
      fill.className = 'contrib__fill contrib__fill--' + dir;
      fill.style.width = barWidth + '%';

      bar.appendChild(fill);

      row.appendChild(label);
      row.appendChild(bar);
      listEl.appendChild(row);
    });
  }
  function renderSummary(result) {
    patientSummary.innerHTML = '';
    const summary = result.patient_summary || {};
    const order = [
      'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
      'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
      'bmi_missing', 'smoking_status',
    ];

    order.forEach(function (key) {
      if (!(key in summary)) return;
      const item = document.createElement('div');
      item.className = 'summary__item';

      const k = document.createElement('span');
      k.className = 'summary__item__k';
      k.textContent = formatLabel(key);

      const v = document.createElement('span');
      v.className = 'summary__item__v';
      const raw = summary[key];
      v.textContent = raw === null || raw === '' ? 'Not provided' : escapeHtml(raw);

      item.appendChild(k);
      item.appendChild(v);
      patientSummary.appendChild(item);
    });
  }

  function renderResult(result) {
    renderPrediction(result);

    const shap = result.shap_explanation || {};
    renderContributions(shapList, shap.contributions, 'value');
    if (!shap.available) {
      shapList.innerHTML =
        '<div class="explainer__unavailable"><p>' +
        escapeHtml(shap.message || 'SHAP explanation unavailable.') + '</p></div>';
    }

    const lime = result.lime_explanation || {};
    renderContributions(limeList, lime.contributions, 'weight');
    if (!lime.available) {
      limeList.innerHTML =
        '<div class="explainer__unavailable"><p>' +
        escapeHtml(lime.message || 'LIME explanation unavailable.') + '</p></div>';
    }

    renderSummary(result);

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function showApiError(message) {
    clearErrors();
    const existing = form.querySelector('.api-error');
    if (existing) existing.remove();

    const banner = document.createElement('div');
    banner.className = 'api-error';
    banner.style =
      'background:#FBEDEE;border:1px solid #F3CBCD;color:#7a2d33;padding:1rem 1.25rem;border-radius:var(--radius-md);margin-bottom:1.5rem;font-size:0.95rem;line-height:1.5;';
    banner.innerHTML =
      '<strong>Assessment could not be completed.</strong><br>' +
      escapeHtml(message) +
      '<br><br>If the model file is missing, add <code>models/strokeguard_catboost.cbm</code> and restart the server.';
    form.prepend(banner);
  }
  // ------------------------------------------------------------------ Form submit
  form.addEventListener('submit', async function (event) {
    event.preventDefault();
    if (submitting) return;

    clearErrors();
    const existingBanner = form.querySelector('.api-error');
    if (existingBanner) existingBanner.remove();

    const formData = new FormData(form);
    const payload = {};

    // Validate gender field
    const genderValue = formData.get('gender');
    if (!genderValue || !'Male Female Other'.includes(genderValue)) {
      showApiError('Please select a valid gender (Male, Female, or Other).');
      resultsSection.style.display = 'none';
      return;
    }
    payload.gender = genderValue;

    // Validate work_type field
    const workTypeValue = formData.get('work_type');
    const allowedWorkTypes = ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'];
    if (!workTypeValue || !allowedWorkTypes.includes(workTypeValue)) {
      showApiError('Please select a valid work type (Private, Self-employed, Govt_job, Children, or Never_worked).');
      resultsSection.style.display = 'none';
      return;
    }
    payload.work_type = workTypeValue;

    // Validate residence_type field
    const residenceTypeValue = formData.get('residence_type');
    const allowedResidenceTypes = ['Urban', 'Rural'];
    if (!residenceTypeValue || !allowedResidenceTypes.includes(residenceTypeValue)) {
      showApiError('Please select a valid residence type (Urban or Rural).');
      resultsSection.style.display = 'none';
      return;
    }
    payload.residence_type = residenceTypeValue;

    payload.age = parseFloat(formData.get('age'));
    payload.hypertension = parseInt(formData.get('hypertension') || '0', 10);
    payload.heart_disease = parseInt(formData.get('heart_disease') || '0', 10);
    payload.ever_married = formData.get('ever_married');
    payload.avg_glucose_level = parseFloat(formData.get('avg_glucose_level'));

    const bmiRaw = formData.get('bmi');
    if (bmiRaw !== null && bmiRaw !== '') {
      payload.bmi = parseFloat(bmiRaw);
    } else {
      payload.bmi = null;
    }

    payload.smoking_status = formData.get('smoking_status');

    showLoading(true);

    try {
      const response = await fetch(API_BASE + '/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const bodyText = await response.text();
      let data;
      try {
        data = JSON.parse(bodyText);
      } catch (e) {
        throw new Error('Invalid response from server.');
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Server error (HTTP ' + response.status + ').');
      }

      renderResult(data);
    } catch (err) {
      console.error('Prediction error:', err);
      showApiError(err.message || 'Prediction could not be completed.');
      resultsSection.style.display = 'none';
    } finally {
      showLoading(false);
    }
  });

  // ------------------------------------------------------------------ Reset
  resetBtn.addEventListener('click', function () {
    if (submitting) return;
    form.reset();
    clearErrors();
    const banner = form.querySelector('.api-error');
    if (banner) banner.remove();
    resultsSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  newAssessmentBtn.addEventListener('click', function () {
    if (submitting) return;
    resultsSection.style.display = 'none';
    form.reset();
    clearErrors();
    const banner = form.querySelector('.api-error');
    if (banner) banner.remove();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // ------------------------------------------------------------------ Health check on load
  fetch(API_BASE + '/api/health')
    .then(function (res) { return res.json(); })
    .then(function (data) {
      if (!data.model_loaded) {
        const banner = document.createElement('div');
        banner.className = 'api-error';
        banner.style =
          'background:#FBF3EC;border:1px solid #E8D9B8;color:#8a541e;padding:1rem 1.25rem;border-radius:var(--radius-md);margin-bottom:1.5rem;font-size:0.9rem;line-height:1.5;';
        banner.innerHTML =
          '<strong>Model not loaded.</strong> ' +
          'The trained CatBoost model file <code>models/strokeguard_catboost.cbm</code> is missing. ' +
          'Add it and restart the server to enable predictions.';
        form.prepend(banner);
      }
    })
    .catch(function () {
      // Silently ignore health check failures; form submit will report the error.
    });
})();
