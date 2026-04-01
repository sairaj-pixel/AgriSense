const App = {
    currentCity: null,
    weatherData: null,
    sidebarMap: null,
    sidebarMarker: null,
    diseaseImageData: null,
    modelStats: null,
    chartInstances: {},
    defaults: { temp: 25, rain: 120, moisture: 40 },
    lastPrediction: null,
    
    init() {
        this.initTabs();
        this.initCitySearch();
        this.initPredictForm();
        this.initDiseaseUpload();
        this.loadModelStats();
        this.loadRecommendations();
    },

    initTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');

                if (btn.dataset.tab === 'model' && !this.modelStats) this.loadModelStats();
                if (btn.dataset.tab === 'recommend') this.loadRecommendations();
                if (btn.dataset.tab === 'weather') this.renderWeatherTab();
            });
        });

        document.getElementById('btn-export-prediction')?.addEventListener('click', () => {
            if (this.lastPrediction) {
                this.exportPrediction('json');
            } else {
                alert('Make a prediction first!');
            }
        });

        document.getElementById('btn-export-rec')?.addEventListener('click', () => {
            this.exportRecommendations('json');
        });
    },

    initCitySearch() {
        const input = document.getElementById('city-input');
        const dropdown = document.getElementById('city-dropdown');
        let debounce;

        input.addEventListener('input', () => {
            clearTimeout(debounce);
            const q = input.value.trim();
            if (q.length < 2) { dropdown.classList.remove('show'); return; }
            debounce = setTimeout(async () => {
                try {
                    const res = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=8&language=en&format=json`);
                    const data = await res.json();
                    
                    if (data.error || !data.results || !data.results.length) {
                        dropdown.classList.remove('show');
                        return;
                    }
                    
                    const cities = data.results.map(c => ({
                        name: c.name || '',
                        admin1: c.admin1 || '',
                        country: c.country || '',
                        lat: c.latitude,
                        lon: c.longitude
                    }));
                    
                    dropdown.innerHTML = cities.map((c, i) =>
                        `<div class="city-option" data-idx="${i}">
                            <div class="city-name">${c.name}</div>
                            <div class="city-sub">${c.admin1 ? c.admin1 + ', ' : ''}${c.country}</div>
                        </div>`
                    ).join('');
                    dropdown.classList.add('show');
                    dropdown._cities = cities;
                } catch { dropdown.classList.remove('show'); }
            }, 300);
        });

        dropdown.addEventListener('click', (e) => {
            const opt = e.target.closest('.city-option');
            if (!opt) return;
            const city = dropdown._cities[parseInt(opt.dataset.idx)];
            this.selectCity(city);
            dropdown.classList.remove('show');
            input.value = city.name;
        });

        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-box')) dropdown.classList.remove('show');
        });
    },

    async selectCity(city) {
        this.currentCity = city;

        const badge = document.getElementById('selected-city');
        document.getElementById('selected-city-text').textContent = `${city.name}, ${city.country}`;
        badge.classList.add('show');

        const mapEl = document.getElementById('sidebar-map');
        mapEl.classList.add('show');
        
        if (!this.sidebarMap) {
            this.sidebarMap = L.map('sidebar-map', { 
                zoomControl: true, 
                attributionControl: false 
            });
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 18
            }).addTo(this.sidebarMap);
        }
        
        if (this.sidebarMarker) this.sidebarMap.removeLayer(this.sidebarMarker);
        this.sidebarMarker = L.marker([city.lat, city.lon]).addTo(this.sidebarMap);
        this.sidebarMap.setView([city.lat, city.lon], 7);
        setTimeout(() => this.sidebarMap.invalidateSize(), 200);

        await this.fetchWeather(city.lat, city.lon);
    },

    async fetchWeather(lat, lon) {
        const weatherEl = document.getElementById('sidebar-weather');
        
        try {
            // Call Open-Meteo directly from browser — avoids Render/server cold-start issues
            const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
                `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,shortwave_radiation_sum` +
                `&hourly=relativehumidity_2m&current_weather=true&timezone=auto&forecast_days=7`;
            const res = await fetch(url);
            const raw = await res.json();

            const daily = raw.daily || {};
            const dates = daily.time || [];
            if (!dates.length) throw new Error('No data');

            // Build daily humidity averages from hourly data
            const hourlyTimes = (raw.hourly || {}).time || [];
            const hourlyRh = (raw.hourly || {}).relativehumidity_2m || [];
            const dailyHumidity = {};
            hourlyTimes.forEach((ht, i) => {
                const day = ht.slice(0, 10);
                if (!dailyHumidity[day]) dailyHumidity[day] = [];
                if (hourlyRh[i] != null) dailyHumidity[day].push(hourlyRh[i]);
            });

            const data = dates.map((d, i) => {
                const tMax = daily.temperature_2m_max[i];
                const tMin = daily.temperature_2m_min[i];
                const temp = Math.round((tMax + tMin) / 2 * 10) / 10;
                const precip = Math.max(0, daily.precipitation_sum[i] || 0);
                const wind = daily.windspeed_10m_max[i] || 0;
                const solar = daily.shortwave_radiation_sum[i] || 0;
                const rhArr = dailyHumidity[d] || [50];
                const humidity = Math.round(rhArr.reduce((a, b) => a + b, 0) / rhArr.length * 10) / 10;
                const moisture = Math.round(Math.min(80, Math.max(10, precip * 0.5 - (temp - 20) * 0.8 + 35)) * 10) / 10;
                const dow = new Date(d + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'long' });
                return {
                    date: d, day_of_week: dow,
                    temp, temp_max: Math.round(tMax * 10) / 10, temp_min: Math.round(tMin * 10) / 10,
                    precip: Math.round(precip * 10) / 10,
                    humidity, wind: Math.round(wind * 10) / 10,
                    moisture, solar_radiation: Math.round(solar * 10) / 10,
                };
            });

            const today = data[0]; // index 0 = today
            document.getElementById('sw-temp').textContent = `${today.temp}°C`;
            document.getElementById('sw-humidity').textContent = `${today.humidity}%`;
            document.getElementById('sw-precip').textContent = `${today.precip} mm`;
            document.getElementById('sw-moisture').textContent = `${today.moisture}%`;
            document.getElementById('sw-wind').textContent = `${today.wind} m/s`;
            document.getElementById('weather-date').textContent = `Date: ${today.date}`;
            weatherEl.classList.add('show');

            document.getElementById('inp-temp').value = today.temp;
            document.getElementById('inp-rain').value = today.precip;
            document.getElementById('inp-moist').value = today.moisture;

            this.weatherData = data;
            this.renderWeatherTab();
            this.loadRecommendations();
        } catch (e) {
            console.error('Weather fetch error:', e);
            weatherEl.classList.remove('show');
            this.weatherData = null;
        }
    },


    initPredictForm() {
        fetch('/api/model-stats').then(r => r.json()).then(data => {
            const cropList = document.getElementById('crop-list');
            
            (data.crop_options || []).forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                cropList.appendChild(opt);
            });
        });

        document.getElementById('btn-predict').addEventListener('click', async () => {
            const btn = document.getElementById('btn-predict');
            btn.disabled = true;
            btn.innerHTML = '<span>⏳</span> Predicting...';
            
            try {
                const temp = parseFloat(document.getElementById('inp-temp').value);
                const rain = parseFloat(document.getElementById('inp-rain').value);
                const ph = parseFloat(document.getElementById('inp-ph').value);
                const fertUsedAcres = parseFloat(document.getElementById('inp-fert-used').value);
                const pestUsedAcres = parseFloat(document.getElementById('inp-pest-used').value);
                const crop = document.getElementById('inp-crop').value;

                // Model expects kg/ha, inputs are kg/acre
                const fertUsed = fertUsedAcres * 2.47105;
                const pestUsed = pestUsedAcres * 2.47105;

                if (isNaN(temp) || isNaN(rain) || isNaN(ph) || isNaN(fertUsedAcres) || isNaN(pestUsedAcres)) {
                    throw new Error('Please enter valid numbers');
                }

                const body = {
                    temperature: temp,
                    rainfall: rain,
                    soil_ph: ph,
                    fertilizer_used: fertUsed,
                    pesticides_used: pestUsed,
                    crop_type: crop,
                };
                
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const data = await res.json();
                
                if (data.error) throw new Error(data.error);

                this.lastPrediction = data;
                
                const resultBox = document.getElementById('predict-result');
                const yieldPerAcre = (data.predicted_yield / 2.47105).toFixed(2);
                document.getElementById('predict-result-text').textContent = yieldPerAcre;
                resultBox.classList.add('show');

                const insightEl = document.getElementById('predict-insight');
                insightEl.innerHTML = `
                    <strong>Input Conditions:</strong><br>
                    🌡️ Temperature: <strong>${temp}°C</strong> | 
                    🌧️ Rainfall: <strong>${rain}mm</strong> | 
                    🧪 Soil pH: <strong>${ph}</strong><br><br>
                    🌾 Crop: <strong>${crop}</strong> | 
                    🚜 Fertilizer: <strong>${fertUsedAcres} kg/acre</strong> | 
                    🛡️ Pesticide: <strong>${pestUsedAcres} kg/acre</strong><br><br>
                    📊 Predicted Yield: <strong>${yieldPerAcre} t/acre</strong><br>
                    📈 Dataset Average: ${(data.mean_yield / 2.47105).toFixed(2)} t/acre<br>
                    📋 Quality: <strong>${data.quality}</strong><br>
                    ${data.confidence_interval ? `<br>📐 Confidence Range: ${(data.confidence_interval.low / 2.47105).toFixed(2)} - ${(data.confidence_interval.high / 2.47105).toFixed(2)} t/acre` : ''}
                `;
            } catch (e) {
                alert('Prediction error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>🎯</span> Predict Yield';
            }
        });
    },

    async loadRecommendations() {
        const temp = parseFloat(document.getElementById('inp-temp').value) || this.defaults.temp;
        const rain = parseFloat(document.getElementById('inp-rain').value) || this.defaults.rain;
        const moisture = parseFloat(document.getElementById('inp-moist').value) || this.defaults.moisture;
        const landSizeAcres = parseFloat(document.getElementById('inp-land-size')?.value) || 2.5;
        const landSize = landSizeAcres / 2.47105;

        const contextEl = document.getElementById('recommend-context');
        if (this.currentCity) {
            contextEl.innerHTML = `
                <span class="icon">📍</span> <strong>${this.currentCity.name}</strong> | 
                <span class="icon">🌡️</span> ${temp}°C | 
                <span class="icon">🌧️</span> ${rain}mm | 
                <span class="icon">💧</span> ${moisture}% |
                <span class="icon">📏</span> ${landSizeAcres.toFixed(1)} acres
            `;
        } else {
            contextEl.innerHTML = `<span class="icon">💡</span> Enter conditions in the Prediction tab or select a city to get personalized recommendations`;
        }

        const spinner = document.getElementById('recommend-spinner');
        const container = document.getElementById('recommend-content');
        
        if (spinner) spinner.classList.add('show');

        try {
            const res = await fetch(`/api/recommend?temp=${temp}&rain=${rain}&moisture=${moisture}&land_size=${landSize}`);
            const data = await res.json();
            
            if (spinner) spinner.classList.remove('show');
            
            if (data.error) {
                container.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><h3>Error</h3><p>${data.error}</p></div>`;
                return;
            }

            const best = data.recommendations[0];
            const conditions = data.conditions;

            let conditionsHtml = conditions.map(c =>
                `<span class="${c.ok ? 'condition-ok' : 'condition-bad'}">${c.ok ? '✅' : '⚠️'} ${c.text}</span>`
            ).join('<br>');

            let cardsHtml = data.recommendations.map((r, i) => {
                const badgeClass = r.suitability >= 60 ? 'high' : (r.suitability >= 35 ? 'medium' : 'low');
                const isTop = i === 0;
                return `
                    <div class="crop-card ${isTop ? 'top-recommendation' : ''}">
                        <div class="crop-emoji">${r.emoji}</div>
                        <div class="crop-info">
                            <div class="crop-name">
                                ${r.crop}
                                ${isTop ? '<span class="top-rec-badge">BEST</span>' : ''}
                            </div>
                            <div class="crop-desc">${r.desc}</div>
                            <div class="crop-detail-grid">
                                <div class="crop-detail-item">
                                    <div class="crop-detail-label">Season</div>
                                    <div class="crop-detail-value">${r.season}</div>
                                </div>
                                <div class="crop-detail-item">
                                    <div class="crop-detail-label">Harvest</div>
                                    <div class="crop-detail-value">${r.days_to_harvest} days</div>
                                </div>
                                <div class="crop-detail-item">
                                    <div class="crop-detail-label">Soil</div>
                                    <div class="crop-detail-value">${r.soil_type}</div>
                                </div>
                                <div class="crop-detail-item" style="grid-column: span 3; background: #f0fdf4;">
                                    <div class="crop-detail-label">🧪 Recommended Fertilizer</div>
                                    <div class="crop-detail-value">${r.fertilizer}</div>
                                </div>
                                <div class="crop-detail-item" style="grid-column: span 3; background: #eff6ff;">
                                    <div class="crop-detail-label">💧 Est. Water Required (${landSizeAcres.toFixed(1)} acres)</div>
                                    <div class="crop-detail-value">${r.water_req_liters ? r.water_req_liters.toLocaleString() + ' Liters' : 'Sufficient Rainfall'}</div>
                                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Assuming ${rain}mm rain. Need ${r.water_needs_mm}mm total.</div>
                                </div>
                            </div>
                        </div>
                        <div class="crop-badge ${badgeClass}">${r.suitability}%</div>
                        <div class="crop-yield">
                            <div class="crop-yield-val">${(r.predicted_yield / 2.47105).toFixed(2)}</div>
                            <div class="crop-yield-label">t/acre</div>
                        </div>
                        <button class="btn btn-outline btn-sm" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="App.viewCropPlan('${r.crop}')">
                            📅 View 7-Day Plan
                        </button>
                    </div>
                `;
            }).join('');

            container.innerHTML = `
                <div class="insight-box" style="margin-bottom:1.25rem;">${conditionsHtml}</div>
                ${cardsHtml}
            `;
        } catch (e) {
            if (spinner) spinner.classList.remove('show');
            container.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><h3>Error</h3><p>Failed to load recommendations</p></div>`;
        }
    },

    async viewCropPlan(crop) {
        const planTabBtn = document.getElementById('tab-btn-plan');
        planTabBtn.style.display = 'flex';
        planTabBtn.click();
        
        const container = document.getElementById('plan-days-container');
        const headerInfo = document.getElementById('plan-header-info');
        const spinner = document.getElementById('plan-spinner');
        
        container.innerHTML = '';
        headerInfo.innerHTML = '';
        spinner.style.display = 'block';
        
        if (!this.currentCity) {
            spinner.style.display = 'none';
            container.innerHTML = `<div class="empty-state"><div class="empty-icon">📍</div><h3>No Location</h3><p>Search for a city in the sidebar first.</p></div>`;
            return;
        }

        const landSizeAcres = parseFloat(document.getElementById('inp-land-size')?.value) || 2.5;

        try {
            const forecastRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${this.currentCity.lat}&longitude=${this.currentCity.lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto`);
            const forecastData = await forecastRes.json();

            const res = await fetch(`/api/crop-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop: crop, land_size: landSizeAcres, forecast: forecastData })
            });
            const data = await res.json();
            
            spinner.style.display = 'none';
            if (data.error) throw new Error(data.error);

            headerInfo.innerHTML = `
                <strong>🌾 Crop:</strong> ${data.crop} &nbsp;|&nbsp; 
                <strong>📍 Location:</strong> ${this.currentCity.name} &nbsp;|&nbsp; 
                <strong>📏 Land:</strong> ${data.land_size_acres} acres
            `;

            container.innerHTML = data.plan.map(p => {
                return `
                    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.25rem; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; gap: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem;">
                            <strong style="font-size: 1.1rem; color: var(--text-primary);">${p.day}</strong> 
                            <span style="color:var(--text-secondary); font-size:0.85rem;">${p.date}</span>
                        </div>
                        
                        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.85rem; color: var(--text-secondary);">🌡️ Temp Range</span>
                                <strong style="font-size: 0.95rem; color: var(--text-primary);">${p.temp_max}°C / ${p.temp_min}°C</strong>
                            </div>

                            <div style="display: flex; justify-content: space-between; align-items: center; ${p.rain_mm > 0 ? 'background: rgba(59,130,246,0.05); padding: 6px 8px; border-radius: 4px; margin: 0 -8px;' : ''}">
                                <span style="font-size: 0.85rem; color: var(--text-secondary);">🌧️ Expected Rain</span>
                                <strong style="font-size: 0.95rem; color: ${p.rain_mm > 0 ? '#3b82f6' : 'var(--text-primary)'};">${p.rain_mm} mm</strong>
                            </div>

                            <div style="background: rgba(16,185,129,0.05); padding: 0.85rem; border-radius: var(--radius-sm); border-left: 3px solid var(--success); margin-top: 0.25rem;">
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem;">💧 Watering Needs</div>
                                <div style="font-weight: 700; color: var(--text-primary); font-size: 1.1rem;">${p.irrigation_liters.toLocaleString()} Liters</div>
                                ${p.irrigation_liters === 0 ? '<div style="font-size:0.75rem; color:var(--success); margin-top:4px; font-weight: 500;">Rain is sufficient!</div>' : ''}
                            </div>

                            <div style="background: ${p.fert_action !== 'None' ? 'rgba(245,158,11,0.05)' : 'rgba(255,255,255,0.03)'}; padding: 0.85rem; border-radius: var(--radius-sm); border-left: 3px solid ${p.fert_action !== 'None' ? 'var(--warning)' : 'var(--border)'};">
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem;">🧪 Fertilizer Action</div>
                                <div style="font-weight: ${p.fert_action !== 'None' ? '600' : '500'}; color: var(--text-primary); font-size: 0.95rem;">${p.fert_action}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

        } catch (e) {
            spinner.style.display = 'none';
            container.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><h3>Plan Error</h3><p>${e.message || 'Failed to fetch schedule.'}</p></div>`;
        }
    },

    renderWeatherTab() {
        const container = document.getElementById('weather-content');
        
        if (!this.weatherData || !this.weatherData.length) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🌍</div>
                    <h3>No Location Selected</h3>
                    <p>Search for a city in the sidebar to view weather data</p>
                </div>
            `;
            return;
        }

        const dates = this.weatherData.map(r => r.date);
        const dayOfWeek = this.weatherData.map(r => r.day_of_week || '');
        const precips = this.weatherData.map(r => r.precip);
        const totalRain = precips.reduce((a, b) => a + b, 0);
        const avgTemp = (this.weatherData.reduce((a, r) => a + r.temp, 0) / this.weatherData.length).toFixed(1);
        const avgHumidity = (this.weatherData.reduce((a, r) => a + r.humidity, 0) / this.weatherData.length).toFixed(0);
        const avgHi = (this.weatherData.reduce((a, r) => a + r.temp_max, 0) / this.weatherData.length).toFixed(1);
        const avgLo = (this.weatherData.reduce((a, r) => a + r.temp_min, 0) / this.weatherData.length).toFixed(1);

        let rainClass, rainText;
        if (totalRain < 5) {
            rainText = `☀️ <b>Dry period:</b> Only ${totalRain.toFixed(1)}mm over 7 days. Consider irrigation.`;
        } else if (totalRain > 100) {
            rainText = `🌧️ <b>Heavy rainfall:</b> ${totalRain.toFixed(1)}mm. Good for water-intensive crops, watch for waterlogging.`;
        } else {
            rainText = `🌤️ <b>Moderate rainfall:</b> ${totalRain.toFixed(1)}mm. Balanced conditions for most crops.`;
        }

        let tableRows = this.weatherData.map(r =>
            `<tr><td>${r.date}<br><span class="day-of-week">${r.day_of_week || ''}</span></td><td>${r.temp}°C</td><td>${r.temp_max}°C</td><td>${r.temp_min}°C</td><td>${r.precip}mm</td><td>${r.humidity}%</td><td>${r.wind}m/s</td><td>${r.moisture}%</td></tr>`
        ).join('');

        container.innerHTML = `
            <div class="weather-summary">
                <div class="weather-summary-card">
                    <div class="wsc-icon">🌧️</div>
                    <div class="wsc-value">${totalRain.toFixed(1)}mm</div>
                    <div class="wsc-label">Total Precipitation</div>
                </div>
                <div class="weather-summary-card">
                    <div class="wsc-icon">🌡️</div>
                    <div class="wsc-value">${avgTemp}°C</div>
                    <div class="wsc-label">Avg Temperature</div>
                </div>
                <div class="weather-summary-card">
                    <div class="wsc-icon">💦</div>
                    <div class="wsc-value">${avgHumidity}%</div>
                    <div class="wsc-label">Avg Humidity</div>
                </div>
                <div class="weather-summary-card">
                    <div class="wsc-icon">📈</div>
                    <div class="wsc-value">${avgHi}°C</div>
                    <div class="wsc-label">Avg High</div>
                </div>
                <div class="weather-summary-card">
                    <div class="wsc-icon">📉</div>
                    <div class="wsc-value">${avgLo}°C</div>
                    <div class="wsc-label">Avg Low</div>
                </div>
            </div>

            <h3 class="section-heading">7-Day Weather Forecast</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem;">
                ${this.weatherData.map(r => `
                    <div class="weather-day-card">
                        <div class="day">${r.day_of_week ? r.day_of_week.slice(0,3) : ''}</div>
                        <div class="date">${r.date.slice(5)}</div>
                        <div class="temp">${r.temp}°C</div>
                        <div class="temp-range">${r.temp_max}° / ${r.temp_min}°</div>
                        <div class="precip">🌧️ ${r.precip}mm</div>
                    </div>
                `).join('')}
            </div>

            <div class="insight-box">${rainText}</div>

            <h3 class="section-heading">Daily Precipitation Chart</h3>
            <div class="chart-wrapper"><canvas id="chart-precip"></canvas></div>

            <h3 class="section-heading">Temperature Range Chart</h3>
            <div class="chart-wrapper"><canvas id="chart-temp-range"></canvas></div>

            <details class="expander">
                <summary>View raw weather data</summary>
                <div class="expander-content">
                    <div class="data-table-wrapper">
                        <table class="data-table">
                            <thead><tr><th>Date</th><th>Avg</th><th>High</th><th>Low</th><th>Precip</th><th>Humidity</th><th>Wind</th><th>Soil Moist.</th></tr></thead>
                            <tbody>${tableRows}</tbody>
                        </table>
                    </div>
                </div>
            </details>
        `;

        this.renderChart('chart-precip', {
            type: 'bar',
            data: {
                labels: dates.map(d => d.slice(5)),
                datasets: [{
                    label: 'Precipitation (mm)',
                    data: precips,
                    backgroundColor: precips.map(p => p < 10 ? '#93c5fd' : (p < 50 ? '#3b82f6' : '#1e40af')),
                    borderRadius: 6,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'mm' } },
                    x: { grid: { display: false } }
                }
            }
        });

        this.renderChart('chart-temp-range', {
            type: 'line',
            data: {
                labels: dates.map(d => d.slice(5)),
                datasets: [
                    {
                        label: 'High',
                        data: this.weatherData.map(r => r.temp_max),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239,68,68,0.1)',
                        fill: '+1',
                        tension: 0.3,
                        pointRadius: 4,
                    },
                    {
                        label: 'Low',
                        data: this.weatherData.map(r => r.temp_min),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.05)',
                        tension: 0.3,
                        pointRadius: 4,
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { title: { display: true, text: '°C' } },
                    x: { grid: { display: false } }
                }
            }
        });
    },

    initDiseaseUpload() {
        const radios = document.querySelectorAll('input[name="disease-method"]');
        const uploadSection = document.getElementById('upload-section');
        const cameraSection = document.getElementById('camera-section');
        const fileInput = document.getElementById('file-input');
        const uploadZone = document.getElementById('upload-zone');
        const previewDiv = document.getElementById('upload-preview');
        const previewImg = document.getElementById('preview-img');
        const analyzeBtn = document.getElementById('btn-analyze');
        let cameraStream = null;

        radios.forEach(r => r.addEventListener('change', (e) => {
            document.querySelectorAll('.method-option').forEach(o => o.classList.remove('active'));
            e.target.closest('.method-option').classList.add('active');
            
            if (e.target.value === 'upload') {
                uploadSection.style.display = '';
                cameraSection.classList.remove('show');
                this.stopCamera(cameraStream);
            } else {
                uploadSection.style.display = 'none';
                cameraSection.classList.add('show');
                this.startCamera(cameraStream);
            }
        }));

        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', e => { 
            e.preventDefault(); 
            uploadZone.classList.add('dragover'); 
        });
        uploadZone.addEventListener('dragleave', () => { 
            uploadZone.classList.remove('dragover'); 
        });
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) this.handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) this.handleFile(fileInput.files[0]);
        });

        this.handleFile = (file) => {
            this.diseaseImageData = file;
            const url = URL.createObjectURL(file);
            previewImg.src = url;
            previewDiv.classList.add('show');
            analyzeBtn.disabled = false;
            cameraSection.classList.remove('show');
            uploadSection.style.display = '';
        };

        document.getElementById('btn-remove-preview').addEventListener('click', () => {
            previewDiv.classList.remove('show');
            this.diseaseImageData = null;
            analyzeBtn.disabled = true;
            fileInput.value = '';
        });

        this.startCamera = async (stream) => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                document.getElementById('camera-video').srcObject = stream;
                cameraStream = stream;
            } catch {
                alert('Could not access camera. Please use upload instead.');
                document.querySelector('input[value="upload"]').checked = true;
                document.querySelectorAll('.method-option')[0].classList.add('active');
                uploadSection.style.display = '';
                cameraSection.classList.remove('show');
            }
        };

        this.stopCamera = (stream) => {
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
            }
        };

        document.getElementById('btn-capture').addEventListener('click', () => {
            const video = document.getElementById('camera-video');
            const canvas = document.getElementById('camera-canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            canvas.toBlob(blob => {
                this.diseaseImageData = blob;
                previewImg.src = URL.createObjectURL(blob);
                previewDiv.classList.add('show');
                analyzeBtn.disabled = false;
                this.stopCamera(cameraStream);
                cameraStream = null;
                cameraSection.classList.remove('show');
                uploadSection.style.display = '';
            }, 'image/jpeg', 0.9);
        });

        analyzeBtn.addEventListener('click', async () => {
            if (!this.diseaseImageData) return;
            
            const spinner = document.getElementById('disease-spinner');
            const resultArea = document.getElementById('disease-result-area');
            spinner.classList.add('show');
            analyzeBtn.disabled = true;

            try {
                const formData = new FormData();
                formData.append('image', this.diseaseImageData);
                const res = await fetch('/api/disease', { method: 'POST', body: formData });
                const data = await res.json();

                spinner.classList.remove('show');
                analyzeBtn.disabled = false;

                if (data.error) throw new Error(data.error);
                if (!data.length) throw new Error('No results');

                const top = data[0];
                const cardClass = top.is_healthy ? 'healthy' : 'sick';
                const statusColor = top.is_healthy ? '#10b981' : '#ef4444';
                const statusText = top.is_healthy ? '✅ Healthy Plant' : '⚠️ Disease Detected';

                let otherResults = '';
                if (data.length > 1) {
                    let otherItems = data.slice(1).map(r =>
                        `<li><b>${r.plant}</b> — ${r.disease} (${r.confidence}%)</li>`
                    ).join('');
                    otherResults = `
                        <details class="expander" style="margin-top:1rem;">
                            <summary>Other possibilities (${data.length - 1})</summary>
                            <div class="expander-content"><ul style="margin-left:1.25rem;">${otherItems}</ul></div>
                        </details>
                    `;
                }

                resultArea.innerHTML = `
                    <div class="disease-result ${cardClass}">
                        <div class="disease-status ${cardClass}">${statusText}</div>
                        <div class="disease-info">
                            <b>Plant:</b> ${top.plant} &nbsp;|&nbsp; <b>Condition:</b> ${top.disease}
                        </div>
                        <div class="disease-meta">
                            Confidence: <b>${top.confidence}%</b> &nbsp;|&nbsp; Severity: <b>${top.severity}</b>
                        </div>
                        <div class="disease-section">
                            <div class="disease-section-title">📋 Description</div>
                            <div class="disease-section-content">${top.description}</div>
                        </div>
                        <div class="disease-section">
                            <div class="disease-section-title">💊 Recommended Treatment</div>
                            <div class="disease-section-content">${top.remedy}</div>
                        </div>
                    </div>
                    ${otherResults}
                `;
                resultArea.classList.add('show');
            } catch (e) {
                spinner.classList.remove('show');
                analyzeBtn.disabled = false;
                resultArea.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">⚠️</div>
                        <h3>Analysis Failed</h3>
                        <p>${e.message || 'The AI service may be loading. Please wait 30 seconds and try again.'}</p>
                    </div>
                `;
            }
        });
    },

    async loadModelStats() {
        try {
            const res = await fetch('/api/model-stats');
            this.modelStats = await res.json();
            this.renderModelTab(this.modelStats);
        } catch {
            document.getElementById('model-metrics').innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">⚠️</div>
                    <h3>Error</h3>
                    <p>Failed to load model statistics</p>
                </div>
            `;
        }
    },

    renderModelTab(data) {
        const metrics = data.metrics || {};
        const modelInfo = data.model_info || {};
        
        const accuracyLevel = data.accuracy_level || 'Moderate';
        const accuracyClass = accuracyLevel.toLowerCase();
        
        document.getElementById('model-metrics').innerHTML = `
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value highlight">${metrics.mae || 0} t/ha</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">${metrics.rmse || 0} t/ha</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value highlight">${metrics.r2_pct || '0%'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAPE</div>
                <div class="metric-value">${metrics.mape || 0}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cross-Val MAE</div>
                <div class="metric-value">${metrics.cv_mae || 0} ± ${metrics.cv_mae_std || 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Training Samples</div>
                <div class="metric-value">${modelInfo.training_samples || 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Samples</div>
                <div class="metric-value">${modelInfo.test_samples || 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">${modelInfo.total_samples || 0}</div>
            </div>
        `;

        document.getElementById('model-accuracy-insight').innerHTML = `
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                <strong>Model Accuracy Level:</strong>
                <span class="accuracy-badge ${accuracyClass}">${accuracyLevel}</span>
            </div>
            <b>What do these metrics mean?</b><br>
            • <b>MAE = ${metrics.mae}</b> — predictions are off by ~${metrics.mae} t/ha on average.<br>
            • <b>RMSE</b> — root mean squared error, penalizes large errors more.<br>
            • <b>R² = ${metrics.r2_pct}</b> — the model explains <b>${Math.round((metrics.r2 || 0) * 100)}%</b> of yield variation.<br>
            • <b>MAPE = ${metrics.mape}%</b> — mean absolute percentage error.<br>
            • <b>Cross-Val MAE</b> — cross-validated error for robustness (${metrics.cv_mae} ± ${metrics.cv_mae_std}).
        `;

        const actual = Array.isArray(data.actual) ? data.actual : [];
        const predicted = Array.isArray(data.predicted) ? data.predicted : [];
        
        const minVal = Math.min(...actual) || 0;
        const maxVal = Math.max(...actual) || 1;
        
        this.renderChart('chart-scatter', {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                        data: actual.map((a, i) => ({ x: a, y: predicted[i] || a })),
                        backgroundColor: 'rgba(16,185,129,0.6)',
                        borderColor: 'rgba(16,185,129,0.8)',
                        pointRadius: 5,
                    },
                    {
                        label: 'Perfect Prediction',
                        data: [
                            { x: minVal, y: minVal },
                            { x: maxVal, y: maxVal }
                        ],
                        type: 'line',
                        borderColor: '#ef4444',
                        borderDash: [6, 4],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: { title: { display: true, text: 'Actual Yield (t/ha)' } },
                    y: { title: { display: true, text: 'Predicted Yield (t/ha)' } }
                }
            }
        });

        document.getElementById('scatter-insight').innerHTML = `
            Each point represents a test sample. The red dashed line indicates perfect predictions.
            Points close to the line are accurate predictions. The spread (std dev) is <b>${data.spread} t/ha</b>.
        `;

        this.renderChart('chart-importance', {
            type: 'bar',
            data: {
                labels: data.importance.map(f => f.feature),
                datasets: [{
                    label: 'Feature Importance',
                    data: data.importance.map(f => Math.abs(f.coefficient)),
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'],
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Importance Score' } },
                    y: { grid: { display: false } }
                }
            }
        });

        const topFeat = data.importance && data.importance.length > 0 ? data.importance[data.importance.length - 1] : null;
        document.getElementById('importance-insight').innerHTML = `
            Feature importance shows which factors have the biggest impact on crop yield predictions.
            <br><strong>Most influential factor:</strong> ${topFeat ? topFeat.feature : 'N/A'}
        `;

        const ys = data.yield_stats;
        const values = Array.isArray(ys.values) ? ys.values : [];
        const bins = this.buildHistogram(values, 12);
        
        this.renderChart('chart-histogram', {
            type: 'bar',
            data: {
                labels: bins.labels,
                datasets: [{
                    label: 'Frequency',
                    data: bins.counts,
                    backgroundColor: '#26a69a',
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Yield (t/ha)' } },
                    y: { title: { display: true, text: 'Number of Samples' }, beginAtZero: true }
                }
            }
        });

        document.getElementById('histogram-insight').innerHTML = `
            The histogram shows how yields are distributed in the training dataset.
            <br>• Minimum: <b>${ys.min}</b> t/ha | Maximum: <b>${ys.max}</b> t/ha
            <br>• Mean: <b>${ys.mean}</b> t/ha | Median: <b>${ys.median}</b> t/ha
            <br>• 50% of data falls between <b>${ys.q25}</b> and <b>${ys.q75}</b> t/ha
        `;

        const residuals = Array.isArray(data.residuals) ? data.residuals : [];
        this.renderChart('chart-residuals', {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Residuals',
                    data: actual.map((a, i) => ({ x: i + 1, y: residuals[i] || 0 })),
                    backgroundColor: residuals.map(r => r >= 0 ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)'),
                    pointRadius: 5,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Sample Index' } },
                    y: { title: { display: true, text: 'Residual (Actual - Predicted)' } }
                }
            }
        });

        const posRes = residuals.filter(r => r > 0).length;
        const negRes = residuals.filter(r => r < 0).length;
        document.getElementById('residuals-insight').innerHTML = `
            Residuals show the difference between actual and predicted values.
            <br>• <b>${posRes}</b> predictions underestimated (positive residuals)
            <br>• <b>${negRes}</b> predictions overestimated (negative residuals)
            <br>• Ideally, residuals should be randomly scattered around zero.
        `;
    },

    renderChart(canvasId, config) {
        if (this.chartInstances[canvasId]) {
            this.chartInstances[canvasId].destroy();
        }
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        this.chartInstances[canvasId] = new Chart(ctx, config);
    },

    buildHistogram(values, numBins) {
        if (!values || values.length === 0) {
            return { labels: ['0'], counts: [0] };
        }
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / numBins || 1;
        const counts = new Array(numBins).fill(0);
        const labels = [];

        for (let i = 0; i < numBins; i++) {
            const lo = min + i * binWidth;
            const hi = lo + binWidth;
            labels.push(`${lo.toFixed(1)}`);
            values.forEach(v => {
                if (v >= lo && (i === numBins - 1 ? v <= hi : v < hi)) counts[i]++;
            });
        }
        return { labels, counts };
    },

    exportPrediction(format) {
        if (!this.lastPrediction) {
            alert('Make a prediction first!');
            return;
        }
        
        const temp = parseFloat(document.getElementById('inp-temp').value);
        const rain = parseFloat(document.getElementById('inp-rain').value);
        const moist = parseFloat(document.getElementById('inp-moist').value);
        const fert = document.getElementById('inp-fertility').value;
        const crop = document.getElementById('inp-crop').value;
        
        const params = new URLSearchParams({
            format: format,
            temperature: temp,
            rainfall: rain,
            soil_moisture: moist,
            fertility: fert,
            crop_type: crop,
            predicted_yield: this.lastPrediction.predicted_yield,
            mean_yield: this.lastPrediction.mean_yield,
            quality: this.lastPrediction.quality,
            confidence_interval_low: this.lastPrediction.confidence_interval?.low || '',
            confidence_interval_high: this.lastPrediction.confidence_interval?.high || ''
        });
        
        window.open(`/api/export-prediction?${params}`, '_blank');
    },

    exportRecommendations(format) {
        const temp = parseFloat(document.getElementById('inp-temp').value) || this.defaults.temp;
        const rain = parseFloat(document.getElementById('inp-rain').value) || this.defaults.rain;
        const moisture = parseFloat(document.getElementById('inp-moist').value) || this.defaults.moisture;
        const landSize = parseFloat(document.getElementById('inp-land-size')?.value) || 1.0;
        
        const params = new URLSearchParams({
            format: format,
            temp: temp,
            rain: rain,
            moisture: moisture,
            land_size: landSize
        });
        
        window.open(`/api/export-recommendations?${params}`, '_blank');
    }
};

window.exportPrediction = (format) => App.exportPrediction(format);

document.addEventListener('DOMContentLoaded', () => App.init());
