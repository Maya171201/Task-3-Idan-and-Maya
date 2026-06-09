import ast
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class MultiLabelTopN(BaseEstimator, TransformerMixin):
    """Encodes multi-label list columns, keeping only the top-N most frequent values per column."""

    def __init__(self, top_n=20):
        self.top_n = top_n

    def fit(self, X, y=None):
        self.top_values_per_col_ = []
        self.mlbs_per_col_ = []
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx] if hasattr(X, 'iloc') else X[:, col_idx]
            col_values = []
            for lst in col_data:
                if isinstance(lst, list):
                    col_values.extend(lst)
            counter = Counter(col_values)
            top_values = [v for v, _ in counter.most_common(self.top_n)]
            self.top_values_per_col_.append(top_values)
            mlb = MultiLabelBinarizer(classes=top_values)
            mlb.fit([top_values])
            self.mlbs_per_col_.append(mlb)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        result_arrays = []
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx] if hasattr(X, 'iloc') else X[:, col_idx]
            top_values = self.top_values_per_col_[col_idx]
            mlb = self.mlbs_per_col_[col_idx]
            filtered = [
                [v for v in lst if v in top_values] if isinstance(lst, list) else []
                for lst in col_data
            ]
            result_arrays.append(mlb.transform(filtered))
        return np.hstack(result_arrays)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f'col{i}' for i in range(self.n_features_in_)]
        names = []
        for col_idx, feat in enumerate(input_features):
            for v in self.top_values_per_col_[col_idx]:
                names.append(f'{feat}={v}')
        return np.array(names)


def prepare_data(df):
    """
    הופך DataFrame גולמי של סרטים לפיצ'רים מוכנים למודל. דטרמיניסטי.
    גישה B: ישויות היסטוריות נשמרות בנפרד (Soviet Union ≠ Russia).
    18 פיצ'רים נקיים ממולטיקולינאריות.
    """

    # ============================================================
    # מילוני נרמול
    # ============================================================
    COUNTRY_MAP = {
        'US': 'United States', 'GB': 'United Kingdom', 'UK': 'United Kingdom',
        'IN': 'India', 'JP': 'Japan', 'DE': 'Germany', 'IT': 'Italy',
        'FR': 'France', 'CA': 'Canada', 'ES': 'Spain', 'RU': 'Russia',
        'MX': 'Mexico', 'KR': 'South Korea', 'KP': 'North Korea',
        'HK': 'Hong Kong', 'CN': 'China', 'AU': 'Australia',
        'NL': 'Netherlands', 'BE': 'Belgium', 'CH': 'Switzerland',
        'AT': 'Austria', 'DK': 'Denmark', 'NO': 'Norway', 'FI': 'Finland',
        'IE': 'Ireland', 'PT': 'Portugal', 'GR': 'Greece',
        'CZ': 'Czech Republic', 'IL': 'Israel', 'IR': 'Iran', 'EG': 'Egypt',
        'TH': 'Thailand', 'ZA': 'South Africa', 'BR': 'Brazil',
        'TR': 'Turkey', 'NZ': 'New Zealand', 'AR': 'Argentina',
        'PL': 'Poland', 'HU': 'Hungary', 'SE': 'Sweden', 'PH': 'Philippines',
        'ID': 'Indonesia', 'MY': 'Malaysia', 'SG': 'Singapore',
        'TW': 'Taiwan', 'VN': 'Vietnam', 'UA': 'Ukraine', 'RO': 'Romania',
        'BG': 'Bulgaria', 'HR': 'Croatia', 'RS': 'Serbia', 'SK': 'Slovakia',
        'SI': 'Slovenia', 'LT': 'Lithuania', 'LV': 'Latvia', 'EE': 'Estonia',
        'IS': 'Iceland', 'PK': 'Pakistan', 'BD': 'Bangladesh',
        'LK': 'Sri Lanka', 'NP': 'Nepal', 'CL': 'Chile', 'CO': 'Colombia',
        'PE': 'Peru', 'VE': 'Venezuela', 'UY': 'Uruguay', 'CU': 'Cuba',
        'CR': 'Costa Rica', 'NG': 'Nigeria', 'KE': 'Kenya', 'MA': 'Morocco',
        'TN': 'Tunisia', 'DZ': 'Algeria', 'SA': 'Saudi Arabia',
        'AE': 'United Arab Emirates', 'QA': 'Qatar', 'LB': 'Lebanon',
        'JO': 'Jordan', 'SY': 'Syria', 'IQ': 'Iraq', 'AF': 'Afghanistan',
        'KZ': 'Kazakhstan', 'GE': 'Georgia', 'AM': 'Armenia',
        'AZ': 'Azerbaijan', 'BY': 'Belarus', 'MD': 'Moldova',
        'LU': 'Luxembourg', 'MT': 'Malta', 'CY': 'Cyprus', 'AL': 'Albania',
        'USA': 'United States', 'U.S.': 'United States',
        'U.S.A.': 'United States', 'United states': 'United States',
        'United State': 'United States',
        'United States of America': 'United States',
        'America': 'United States', 'Great Britain': 'United Kingdom',
        'England': 'United Kingdom', 'Scotland': 'United Kingdom',
        'Wales': 'United Kingdom', 'Northern Ireland': 'United Kingdom',
        'Britain': 'United Kingdom', 'Türkiye': 'Turkey',
        'Turkiye': 'Turkey', 'Czechia': 'Czech Republic',
        'Korea': 'South Korea', 'South Korean': 'South Korea',
        'Hong-Kong': 'Hong Kong', 'Mainland China': 'China',
        "People's Republic of China": 'China',
        'Macedonia': 'North Macedonia', 'Bosnia': 'Bosnia and Herzegovina',
        'Russian Federation': 'Russia', 'México': 'Mexico',
        'USSR': 'Soviet Union', 'Soviet Russia': 'Soviet Union',
        'SFR Yugoslavia': 'Yugoslavia', 'FR Yugoslavia': 'Yugoslavia',
        'FPR Yugoslavia': 'Yugoslavia',
        'Federal Republic of Yugoslavia': 'Yugoslavia',
        'British Hong Kong': 'Hong Kong', 'Republic of China': 'Taiwan',
        'German Reich': 'Nazi Germany',
        'German Democratic Republic': 'East Germany',
        'Weimar Germany': 'Weimar Republic',
        'Polish PR': "Polish People's Republic",
        'Republic of Macedonia': 'North Macedonia',
        'SR Macedonia': 'North Macedonia',
        'Bohemia and Moravia': 'Protectorate of Bohemia and Moravia',
    }

    LANGUAGE_MAP = {
        'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
        'it': 'Italian', 'ja': 'Japanese', 'hi': 'Hindi', 'ru': 'Russian',
        'pt': 'Portuguese', 'ar': 'Arabic', 'zh': 'Mandarin', 'ko': 'Korean',
        'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'bn': 'Bengali',
        'mr': 'Marathi', 'gu': 'Gujarati', 'pa': 'Punjabi', 'ur': 'Urdu',
        'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
        'no': 'Norwegian', 'da': 'Danish', 'fi': 'Finnish', 'cs': 'Czech',
        'el': 'Greek', 'he': 'Hebrew', 'th': 'Thai', 'vi': 'Vietnamese',
        'id': 'Indonesian', 'ro': 'Romanian', 'hu': 'Hungarian',
        'English Language': 'English', 'Standard Mandarin Chinese': 'Mandarin',
        'Mandarin Chinese': 'Mandarin', 'Standard Mandarin': 'Mandarin',
        'Chinese': 'Mandarin', 'Spanish Language': 'Spanish',
        'French Language': 'French', 'Hindi Language': 'Hindi',
        'Bangla': 'Bengali', 'Sinhalese': 'Sinhala', 'Pashtu': 'Pashto',
        'Bahasa Indonesia': 'Indonesian', 'Cantonese Chinese': 'Cantonese',
        'en-US': 'English', 'en-GB': 'English', 'en-IN': 'English',
        'pt-br': 'Portuguese', 'pt-BR': 'Portuguese', 'fr-CA': 'French',
        'Brazilian Portuguese': 'Portuguese', 'Modern Hebrew': 'Hebrew',
        'Old English': 'English', 'Hindustani': 'Hindi',
        'Hindi-Urdu': 'Hindi', 'Hinglish': 'English', 'Deutsch': 'German',
        'Te Reo Māori': 'Maori', 'Māori': 'Maori', 'Sámi': 'Sami',
        'Northern Sámi': 'Sami', 'Northern Sami': 'Sami',
        'Guaraní': 'Guarani', 'Mòoré': 'Mooré', 'Wu Chinese': 'Wu',
        'Egyptian Arabic': 'Arabic', 'Lebanese Arabic': 'Arabic',
        'Moroccan Arabic': 'Arabic', 'Levantine Arabic': 'Arabic',
        'Iraqi Arabic': 'Arabic', 'Algerian Arabic': 'Arabic',
        'Syrian Arabic': 'Arabic', 'Egyptian': 'Arabic',
        'Moroccan': 'Arabic', 'Lebanese': 'Arabic', 'Algerian': 'Arabic',
        'Min Nan': 'Hokkien', 'Taiwanese Hokkien': 'Hokkien',
        'Taiwanese': 'Hokkien', 'Swiss German': 'German',
        'Sound film': 'Silent', 'Silent film': 'Silent',
        'Sound': 'Silent', 'Synchronized': 'Silent',
        'Serbo-Croat': 'Serbo-Croatian', 'Slovene': 'Slovenian',
    }

    KNOWN_COUNTRIES = {
        'United States', 'United Kingdom', 'New Zealand', 'South Africa',
        'South Korea', 'North Korea', 'Hong Kong', 'Czech Republic',
        'Saudi Arabia', 'United Arab Emirates', 'Sri Lanka', 'Costa Rica',
        'Puerto Rico', 'Bosnia and Herzegovina', 'North Macedonia',
        'Dominican Republic', 'Trinidad and Tobago', 'Papua New Guinea',
        'El Salvador', 'Cape Verde', 'Ivory Coast', 'Burkina Faso',
        'Sierra Leone', 'Central African Republic',
        'Democratic Republic of Congo', 'Equatorial Guinea',
        'San Marino', 'Vatican City', 'East Timor',
        'Isle of Man', 'Faroe Islands', 'Cayman Islands',
        'Soviet Union', 'East Germany', 'West Germany',
        'Weimar Republic', 'Nazi Germany', 'Russian Empire',
        'Czechoslovakia', 'Yugoslavia', 'Serbia and Montenegro',
        'British India', 'Empire of Japan', 'Ottoman Empire',
        'Austria-Hungary', 'Austrian Empire',
        "Polish People's Republic", "Mongolian People's Republic",
        'Protectorate of Bohemia and Moravia', 'Vichy France',
        'German Empire', 'Soviet Occupation Zone', 'East Pakistan',
        'Mandatory Palestine', 'Republic of China',
        'India', 'Japan', 'Germany', 'Italy', 'France', 'Canada',
        'Spain', 'Russia', 'Mexico', 'China', 'Australia',
        'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Denmark',
        'Norway', 'Finland', 'Ireland', 'Portugal', 'Greece', 'Israel',
        'Iran', 'Egypt', 'Thailand', 'Brazil', 'Turkey', 'Argentina',
        'Poland', 'Hungary', 'Sweden', 'Philippines', 'Indonesia',
        'Malaysia', 'Singapore', 'Taiwan', 'Vietnam', 'Ukraine',
        'Romania', 'Bulgaria', 'Croatia', 'Serbia', 'Slovakia',
        'Slovenia', 'Lithuania', 'Latvia', 'Estonia', 'Iceland',
        'Pakistan', 'Bangladesh', 'Nepal', 'Chile', 'Colombia',
        'Peru', 'Venezuela', 'Uruguay', 'Cuba', 'Nigeria', 'Kenya',
        'Morocco', 'Tunisia', 'Algeria', 'Qatar', 'Lebanon', 'Jordan',
        'Syria', 'Iraq', 'Afghanistan', 'Kazakhstan', 'Georgia',
        'Armenia', 'Azerbaijan', 'Belarus', 'Moldova', 'Luxembourg',
        'Malta', 'Cyprus', 'Albania', 'Senegal', 'Ghana', 'Cameroon',
        'Uganda', 'Tanzania', 'Mozambique', 'Angola', 'Mongolia',
        'Cambodia', 'Myanmar', 'Laos', 'Bhutan', 'Maldives', 'Suriname',
        'Bolivia', 'Ecuador', 'Paraguay', 'Guatemala', 'Honduras',
        'Panama', 'Jamaica', 'Bahamas', 'Kosovo', 'Montenegro',
        'Liechtenstein', 'Andorra', 'Monaco', 'Greenland', 'Palestine',
        'Yemen', 'Oman', 'Bahrain', 'Kuwait', 'Sudan', 'Ethiopia',
        'Somalia', 'Rwanda', 'Burundi', 'Madagascar', 'Mauritius',
        'Zimbabwe', 'Zambia', 'Botswana', 'Namibia', 'Lesotho',
        'Eswatini', 'Mali', 'Niger', 'Chad', 'Benin', 'Togo', 'Liberia',
        'Gambia', 'Mauritania', 'Gabon', 'Congo', 'Eritrea', 'Djibouti',
        'Comoros', 'Seychelles', 'Fiji', 'Samoa', 'Tonga', 'Haiti',
        'Nicaragua', 'Libya', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',
        'Kyrgyzstan', 'Brunei', 'Macau', 'Bermuda', 'Malawi',
    }

    KNOWN_LANGUAGES = {
        'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
        'Russian', 'Japanese', 'Mandarin', 'Cantonese', 'Korean',
        'Arabic', 'Hindi', 'Bengali', 'Tamil', 'Telugu', 'Marathi',
        'Gujarati', 'Punjabi', 'Urdu', 'Malayalam', 'Kannada', 'Dutch',
        'Swedish', 'Norwegian', 'Danish', 'Finnish', 'Polish', 'Czech',
        'Slovak', 'Hungarian', 'Romanian', 'Bulgarian', 'Greek',
        'Turkish', 'Hebrew', 'Persian', 'Thai', 'Vietnamese',
        'Indonesian', 'Filipino', 'Tagalog', 'Malay', 'Burmese',
        'Khmer', 'Lao', 'Mongolian', 'Tibetan', 'Nepali', 'Sinhala',
        'Pashto', 'Dari', 'Kurdish', 'Armenian', 'Georgian',
        'Azerbaijani', 'Kazakh', 'Uzbek', 'Tajik', 'Kyrgyz', 'Turkmen',
        'Belarusian', 'Ukrainian', 'Serbian', 'Croatian', 'Bosnian',
        'Slovenian', 'Macedonian', 'Albanian', 'Maltese', 'Catalan',
        'Basque', 'Galician', 'Welsh', 'Irish', 'Icelandic', 'Faroese',
        'Estonian', 'Latvian', 'Lithuanian', 'Yiddish', 'Latin',
        'Swahili', 'Zulu', 'Xhosa', 'Afrikaans', 'Amharic', 'Yoruba',
        'Hausa', 'Igbo', 'Wolof', 'Bhojpuri', 'Assamese', 'Odia',
        'Konkani', 'Sanskrit', 'Tatar', 'Quechua', 'Aymara', 'Guarani',
        'Silent', 'Hokkien', 'Hakka', 'Esperanto', 'Maori', 'Hawaiian',
        'Samoan', 'Tongan', 'Fijian', 'Inuktitut', 'Cree', 'Navajo',
        'Cherokee', 'Lakota', 'Mohawk', 'Romansh', 'Sami', 'Frisian',
        'Luxembourgish', 'Flemish', 'Sicilian', 'Neapolitan',
        'Venetian', 'Aramaic', 'Shanghainese', 'Teochew',
        'Cebuano', 'Hiligaynon', 'Ilocano', 'Javanese', 'Sundanese',
        'Balinese', 'Acehnese', 'Dzongkha', 'Maithili', 'Manipuri',
        'Kashmiri', 'Berber', 'Somali', 'Tigrinya', 'Romani',
        'Tahitian', 'Greenlandic', 'Nahuatl', 'Mapudungun', 'Wu',
        'Scots', 'Bashkir', 'Chechen', 'Ossetian', 'Mooré', 'Bambara',
        'Lingala', 'Kinyarwanda', 'Serbo-Croatian',
    }

    KNOWN_GENRES = {
        'Action', 'Adult', 'Adventure', 'Animation', 'Biography',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
        'Music', 'Musical', 'Mystery', 'News', 'Reality-Tv',
        'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
        'Thriller', 'War', 'Western',
    }

    # ============================================================
    # פונקציות עזר
    # ============================================================
    sequel_pattern = re.compile(
        r'\b(II|III|IV|V|VI|VII|VIII|IX|X)\b'
        r'|\bPart\s+\d+\b'
        r'|:\s*\w.*\b[2-9]\b\s*$'
        r'|\b\w+\s+[2-9]\b\s*$',
        flags=re.IGNORECASE
    )

    def is_sequel(title):
        if not isinstance(title, str) or not title:
            return False
        return bool(sequel_pattern.search(title))

    def clean_country(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).strip()
        if not s or s in ('Not Found', 'nan', 'None', ''):
            return []
        s = re.sub(r'\[\s*\d+\s*\]', '', s).strip()
        s = re.sub(r'\[\s*[a-z]+\s*\]', '', s).strip()
        if s.startswith('['):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    s = ', '.join(str(x).strip() for x in parsed)
            except (ValueError, SyntaxError):
                s = re.sub(r"[\[\]'\"]", '', s)
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
        parts = re.split(r',|/|\||  +', s)
        parts = [p.strip() for p in parts if p.strip()]
        result_list = []
        for part in parts:
            mapped = COUNTRY_MAP.get(part, part)
            if mapped in KNOWN_COUNTRIES:
                result_list.append(mapped)
                continue
            words = part.split()
            i = 0
            while i < len(words):
                matched = False
                for window_size in range(5, 0, -1):
                    if i + window_size > len(words):
                        continue
                    candidate = ' '.join(words[i:i + window_size])
                    mapped_candidate = COUNTRY_MAP.get(candidate, candidate)
                    if mapped_candidate in KNOWN_COUNTRIES:
                        result_list.append(mapped_candidate)
                        i += window_size
                        matched = True
                        break
                if not matched:
                    i += 1
        seen = set()
        unique = []
        for item in result_list:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def clean_language(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).strip()
        if not s or s in ('Not Found', 'nan', 'None', ''):
            return []
        s = re.sub(r'\[\s*\d+\s*\]', '', s).strip()
        s = re.sub(r'\[\s*[a-z]+\s*\]', '', s).strip()
        if s.startswith('['):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    s = ', '.join(str(x).strip() for x in parsed)
            except (ValueError, SyntaxError):
                s = re.sub(r"[\[\]'\"]", '', s)
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
        parts = re.split(r',|/|\||  +', s)
        parts = [p.strip() for p in parts if p.strip()]
        result_list = []
        for part in parts:
            mapped = LANGUAGE_MAP.get(part, part)
            if mapped in KNOWN_LANGUAGES:
                result_list.append(mapped)
                continue
            words = part.split()
            i = 0
            while i < len(words):
                matched = False
                for window_size in range(5, 0, -1):
                    if i + window_size > len(words):
                        continue
                    candidate = ' '.join(words[i:i + window_size])
                    mapped_candidate = LANGUAGE_MAP.get(candidate, candidate)
                    if mapped_candidate in KNOWN_LANGUAGES:
                        result_list.append(mapped_candidate)
                        i += window_size
                        matched = True
                        break
                if not matched:
                    i += 1
        seen = set()
        unique = []
        for item in result_list:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def clean_genres(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).strip()
        if not s or s.lower() in ('\\n', '[]', 'nan', 'null', 'none'):
            return []
        if s.startswith('['):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    s = ','.join(str(x) for x in parsed)
            except (ValueError, SyntaxError):
                s = re.sub(r"[\[\]'\"]", '', s)
        parts = s.split(',')
        result_list = []
        seen = set()
        for g in parts:
            g = re.sub(r'[^a-zA-Z\- ]', '', g).strip().title()
            if g in KNOWN_GENRES and g not in seen:
                seen.add(g)
                result_list.append(g)
        return result_list

    def parse_actor_ids(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s or s in ('[]', 'nan', 'None'):
                return []
            try:
                parsed = ast.literal_eval(s)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
        return []

    def count_oscars_before(person_ids, year, oscar_cache):
        if year is None or pd.isna(year) or not person_ids:
            return np.nan, np.nan
        year = int(year)
        per_person = []
        for pid in person_ids:
            wins = oscar_cache.get(pid, [])
            c = sum(1 for w in wins if w.get('year') is not None and w['year'] < year)
            per_person.append(c)
        if not per_person:
            return np.nan, np.nan
        return int(sum(per_person)), int(max(per_person))

    def count_recent_movies(person_ids, current_year, actor_movies_cache, years_back=5):
        if current_year is None or pd.isna(current_year) or not person_ids:
            return np.nan, np.nan
        current_year = int(current_year)
        cutoff = current_year - years_back
        counts = []
        for pid in person_ids:
            movies = actor_movies_cache.get(pid, [])
            c = sum(1 for tid, yr in movies if cutoff <= yr < current_year)
            counts.append(c)
        if not counts:
            return np.nan, np.nan
        return int(sum(counts)), int(max(counts))

    # ============================================================
    # לוגיקה ראשית
    # ============================================================
    df = df.copy().reset_index(drop=True)

    required_cols = ['tconst', 'primaryTitle', 'startYear', 'runtimeMinutes',
                     'genres', 'Language', 'Country', 'lead_actors_ids']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # STEP 1: הסרת leakage
    leakage_cols = ['averageRating', 'numVotes', 'BoxOffice']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # STEP 2: ניקוי נומריות
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
    df.loc[(df['startYear'] < 1895) | (df['startYear'] > 2024), 'startYear'] = np.nan

    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df.loc[(df['runtimeMinutes'] < 60) | (df['runtimeMinutes'] > 300), 'runtimeMinutes'] = np.nan

    # STEP 3: טעינת caches
    if os.path.exists('imdb_enrich_cache.pkl'):
        imdb_cache = pd.read_pickle('imdb_enrich_cache.pkl')
    else:
        imdb_cache = {}

    if os.path.exists('oscar_wins_cache.pkl'):
        oscar_cache = pd.read_pickle('oscar_wins_cache.pkl')
    else:
        oscar_cache = {}

    if os.path.exists('actor_movies_cache.pkl'):
        actor_movies_cache = pd.read_pickle('actor_movies_cache.pkl')
    else:
        actor_movies_cache = {}

    def imdb_get(tconst, key):
        if pd.isna(tconst):
            return None
        rec = imdb_cache.get(tconst)
        return rec.get(key) if rec else None

    # STEP 4: השלמת Language/Country + ניקוי כ-multi-label
    imdb_lang = df['tconst'].apply(lambda t: imdb_get(t, 'language'))
    imdb_country = df['tconst'].apply(lambda t: imdb_get(t, 'country'))
    df['Language'] = df['Language'].where(df['Language'].notna(), imdb_lang)
    df['Country'] = df['Country'].where(df['Country'].notna(), imdb_country)

    df['Country'] = df['Country'].apply(clean_country)
    df['Language'] = df['Language'].apply(clean_language)
    df['genres'] = df['genres'].apply(clean_genres)

    # STEP 5: שם הבמאי + has_director
    df['director_name'] = df['tconst'].apply(lambda t: imdb_get(t, 'director_name'))
    df['has_director'] = df['director_name'].notna().astype(int)

    # STEP 6: num_actors
    actor_id_lists = df['lead_actors_ids'].apply(parse_actor_ids)
    df['num_actors'] = actor_id_lists.apply(len)
    df['num_actors'] = df['num_actors'].replace(0, np.nan)

    # STEP 7: פיצ'רי אוסקר
    director_ids = df['tconst'].apply(lambda t: imdb_get(t, 'director_id'))
    dir_wins = []
    for d, yr in zip(director_ids, df['startYear']):
        s, m = count_oscars_before([d] if d else [], yr, oscar_cache)
        dir_wins.append(s)
    df['director_oscar_wins'] = dir_wins

    act_sum, act_max = [], []
    for ids, yr in zip(actor_id_lists, df['startYear']):
        s, m = count_oscars_before(ids, yr, oscar_cache)
        act_sum.append(s)
        act_max.append(m)
    df['lead_actors_oscar_wins'] = act_sum
    df['lead_actors_oscar_wins_max'] = act_max

    # STEP 8: actors_recent_movies
    all_people = [
        (list(ids) + ([d] if d else []))
        for ids, d in zip(actor_id_lists, director_ids)
    ]
    recent_sum, recent_max = [], []
    for people, yr in zip(all_people, df['startYear']):
        s, m = count_recent_movies(people, yr, actor_movies_cache, years_back=5)
        recent_sum.append(s)
        recent_max.append(m)
    df['actors_recent_movies_sum'] = recent_sum
    df['actors_recent_movies_max'] = recent_max

    # STEP 9: פיצ'רים הנדסיים בסיסיים
    df['screen_time_per_actor'] = df['runtimeMinutes'] / df['num_actors']
    df['screen_time_per_actor'] = df['screen_time_per_actor'].replace(
        [np.inf, -np.inf], np.nan
    )
    df['num_genres'] = df['genres'].apply(len)
    df.loc[df['num_genres'] == 0, 'num_genres'] = np.nan

    df['is_sequel'] = (
        df['primaryTitle'].fillna('').astype(str)
        .apply(is_sequel).astype(int)
    )
    df['decade'] = (df['startYear'] // 10 * 10)

    # STEP 10: Missing Indicator Features
    df['missing_actors'] = df['num_actors'].isna().astype(int)
    df['missing_language'] = df['Language'].apply(
        lambda x: 1 if (not isinstance(x, list) or len(x) == 0) else 0
    )
    df['missing_country'] = df['Country'].apply(
        lambda x: 1 if (not isinstance(x, list) or len(x) == 0) else 0
    )

    # STEP 11: פיצ'רים בינאריים מסכמים
    df['has_oscar_winner'] = (
        (pd.Series(act_max).fillna(0) > 0) |
        (pd.Series(dir_wins).fillna(0) > 0)
    ).astype(int)

    df['actors_very_active'] = (
        pd.Series(recent_max).fillna(0) >= 10
    ).astype(int)

    # STEP 12: הסרת עמודות שלא בשימוש
    cols_to_drop = ['primaryTitle', 'lead_actors_ids', 'budget', 'plot']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # STEP 13: anti-leakage check
    forbidden = {'averageRating', 'numVotes', 'BoxOffice'}
    assert forbidden.isdisjoint(df.columns), \
        f"Leakage detected: {forbidden & set(df.columns)}"

    return df
