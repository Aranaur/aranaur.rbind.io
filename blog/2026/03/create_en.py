import os

with open('sakura_viz.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace texts
content = content.replace('["25 бер.", "1 квіт.", "10 квіт.", "20 квіт.", "1 трав."]', '["Mar 25", "Apr 1", "Apr 10", "Apr 20", "May 1"]')

content = content.replace('Середньовічне потепління', 'Medieval Climate Anomaly')
content = content.replace('Мала льодовикова епоха', 'Little Ice Age')
content = content.replace('Антропогенне потепління', 'Anthropogenic Warming')

content = content.replace(
    'UA_MONTHS = {1:"січ.",2:"лют.",3:"бер.",4:"квіт.",\n             5:"трав.",6:"черв.",7:"лип.",8:"серп.",\n             9:"вер.",10:"жовт.",11:"лист.",12:"груд."}',
    'UA_MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",\n             5:"May",6:"Jun",7:"Jul",8:"Aug",\n             9:"Sep",10:"Oct",11:"Nov",12:"Dec"}'
)

content = content.replace('{mean_dt.day} {UA_MONTHS[mean_dt.month]} — середнє за {YEAR_SPAN} р.', '{UA_MONTHS[mean_dt.month]} {mean_dt.day} — {YEAR_SPAN}-yr mean')
content = content.replace('— найраніше ({dt_earliest.day} {UA_MONTHS[dt_earliest.month]})', '— earliest ({UA_MONTHS[dt_earliest.month]} {dt_earliest.day})')
content = content.replace('— найпізніше ({dt_latest.day} {UA_MONTHS[dt_latest.month]})', '— latest ({UA_MONTHS[dt_latest.month]} {dt_latest.day})')
content = content.replace('f"{yr_min} ({dt_min.day} {UA_MONTHS[dt_min.month]})"', 'f"{yr_min} ({UA_MONTHS[dt_min.month]} {dt_min.day})"')
content = content.replace('f"{yr_max} ({dt_max.day} {UA_MONTHS[dt_max.month]})"', 'f"{yr_max} ({UA_MONTHS[dt_max.month]} {dt_max.day})"')

content = content.replace(
    'Кожна точка — зафіксована дата\n'
    'пікового цвітіння сакури у Кіото.\n'
    'Колір: відхилення від {YEAR_SPAN}-річного\n'
    'середнього ({mean_dt.day} {UA_MONTHS[mean_dt.month]}).',
    'Each point is a recorded date\n'
    'of peak sakura bloom in Kyoto.\n'
    'Color: deviation from the {YEAR_SPAN}-year\n'
    'mean ({UA_MONTHS[mean_dt.month]} {mean_dt.day}).'
)

content = content.replace('−12 дн.\n(раніше)', '−12 days\n(earlier)')
content = content.replace('середнє', 'mean')
content = content.replace('+12 дн.\n(пізніше)', '+12 days\n(later)')
content = content.replace('Відхил.', 'Dev.')

content = content.replace('f"{c} ст."', 'f"{c} C."')
content = content.replace('Дата цвітіння', 'Bloom Date')
content = content.replace('Квітковий календар: зсув масового цвітіння', 'Floral Calendar: shift in mass bloom')
content = content.replace('Частота (%)', 'Frequency (%)')

content = content.replace('Середня температура квітня (°C)', 'April mean temperature (°C)')
content = content.replace('Температурний дощ: потепління по 30-річчях', 'Temperature Rain: warming by 30-yr periods')

content = content.replace('Сакура — 1 200 років весни в Кіото', 'Sakura — 1,200 Years of Spring in Kyoto')
content = content.replace(
    'Дати цвітіння сакури з 812 по 2026 рік демонструють чіткий зсув до ранішого цвітіння внаслідок потепління клімату',
    'Sakura bloom dates from 812 to 2026 demonstrate a clear shift towards earlier blooming due to climate warming'
)

content = content.replace('featured.jpg', 'featured_en.jpg')
content = content.replace('sakura_plot.png', 'sakura_plot_en.png')

with open('sakura_viz_en.py', 'w', encoding='utf-8') as f:
    f.write(content)
