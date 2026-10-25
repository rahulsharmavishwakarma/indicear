# Loanword normalisation (native-script English -> Latin)

Research note, 24 October 2026. Code: `indicear/loanwords.py`; resources:
`indicear/resources/loanwords/` and `indicear/resources/english_words.tsv`.

## Why

IndicEAR's convention is Indic words in native script and English words in Latin. IndicVoices and
Vaani follow it after the gloss rewrite (`markup.py`: "ಆರ್ಡರ್ [order]" -> "order"), but Kathbath,
Shrutilipi and other native-only sources write the same English loanwords in native script
("ಆರ್ಡರ್"). The model is therefore taught both spellings for the same sound. In the pilot this
showed up as Kannada script errors of 12-14% on Kathbath and mid-word script mixing.
`LoanwordNormaliser` rewrites the native-only sources' loanwords to Latin in **training targets**,
so every source teaches one convention. The original text stays available as a `refs`
alternative for script-normalised scoring (the integration into `markup`/`ingest`/`stream` is
documented in docs/framework.md).

## Method

`LoanwordNormaliser.rewrite(text, lang) -> (text, count)` replaces whole native-script words
that are in the language's mapping; every other word is untouched, the word count is kept and
the output stays in `indicear.text.normalize` form. English and unknown languages pass through.

1. **Mined lexicon (the only path enabled).** `python -m indicear.loanwords build` reads IndicVoices
   `unsanitized_normalized` transcripts, where annotators follow an English loanword with its
   Latin gloss (`native [english]`; Table-13 tags such as `[breathing]` are recognised with
   `markup.is_indicvoices_tag` and skipped; multi-word glosses are not used). A native word ->
   English word pair is kept only when all of these hold:

   | Check | Threshold | Purpose |
   | --- | --- | --- |
   | pair seen | >= 3 times (`min_count`) | no one-off annotator choices |
   | dominant gloss | >= 90% of the native word's glosses (`min_dominance`) | one spelling per word |
   | glossed share | >= 80% of the native word's IndicVoices occurrences (`min_gloss_share`) | native words that merely sound English (कमरा "room" vs camera, बस "enough" vs bus) are rarely glossed, so they are rejected |
   | length | >= 3 letters on both sides (`min_letters`, `min_english_letters`) | short words and initialisms ("MM", "TV") are too ambiguous |
   | common English | SCOWL size <= 50 (`max_word_level`) or in `english_extra` | proper names (countries, people, Indian places) stay native; months, weekdays, "english" and a short list of brands and apps people type in Latin are allowed explicitly |
   | English blocklist | `english_block` | common-word homographs of country names (japan, china, turkey, ...) |
   | sounds alike | `markup.sounds_alike` | a translation gloss (காலு / stone) is not a loanword |
   | whole word | consonant skeleton of the whole native word within 0.34 edits per class (exact for 1-2 classes) | a glued case suffix ("ಕಾರ್ನಲ್ಲಿ [car]") would be lost by the rewrite |

2. **Word-list path (implemented, disabled everywhere).** For native words absent from the
   lexicon, a candidate English word with the same consonant skeleton (SCOWL size <= 35), accepted
   by `sounds_alike`, a vowel-aware distance <= 0.25 and a unique best match. On Kathbath words
   never seen in IndicVoices its precision was about 40% (hi: बिसौरी -> fishery, सुपारी -> safari,
   चिन्तन -> suntan; ta: செய்கிறார் -> securer; kn: ಸರಾಸರಿ -> surgery), far below the 95% bar, so
   `params.json` enables it for no language. It is kept for experiments only.
3. **Allowlist and blocklist** (`allow.json`: `{lang: {native: english}}`, forced; `block.json`:
   `{lang: [native, ...]}`, never rewritten). The blocklist holds every wrong rewrite found in the
   review below.

**Identity.** `fingerprint()` is a sha256 over the resource files and the effective parameters;
it changes whenever the lexicon, the lists or the thresholds change, so it belongs in any run
config that uses the normaliser. `describe()` reports the version, parameters, entries per
language and the word-list licence.

## Sources and licences

- **IndicVoices** (`ai4bharat/IndicVoices@c96f9088`, CC BY 4.0): the `unsanitized_normalized`
  column of the train split only, read with HTTP range requests (no audio). hi (12 shards) and ta
  (20) from the pilot's local shards, kn from 5 local + 9 remote shards, the other languages from 14
  random train shards each: 48,000-78,000 transcripts per language, about 230 MB downloaded in all
  (text columns of IndicVoices and Kathbath, plus the 2.6 MB SCOWL tarball).
- **SCOWL** 2020.12.07 (http://wordlist.aspell.net/), `english`, `american`, `british` and
  `canadian` word lists of sizes <= 50, lowercase a-z only, 83,052 words with their size
  (`english_words.tsv`). Licence: the SCOWL copyright notice, a permissive MIT-like grant ("Permission
  to use, copy, modify, distribute and sell these word lists ... for any purpose is hereby granted
  without fee, provided that the above copyright notice appears in all copies"); its component lists
  are public domain or under similar permissive notices. The full text is
  `indicear/resources/SCOWL-LICENSE.txt`, which must ship with the list. wordfreq was not used:
  its data is CC BY-SA 4.0.
- **Kathbath** (`ai4bharat/Kathbath@5b9e9284`, CC BY 4.0): the `text` column of a few train
  shards per language, for the review only.

## Precision review

Kathbath train text (a native-only source) normalised with the shipped lexicon and parameters,
**before** the allowlist and blocklist (review state fingerprint `760402c2…`; the shipped resources,
with the fixes, are `8c57499c…`). 100 rewrites per language were drawn at random (seed `review:<lang>`)
from all rewrite instances and each was judged by hand: is the native word the English word, and is
the Latin spelling right? Every verdict, with its context, is in
`indicear/resources/loanwords/review/<lang>.json` (`python -m indicear.loanwords review` reproduces
the samples exactly from the same text).

| Lang | Lexicon entries | Kathbath words | Rewrites per 1,000 words | Rewritten types | Precision (100 sampled) | Wrong |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hi | 2,062 | 268,266 | 26.6 | 949 | **99%** | उबर (verb "recover") -> uber |
| mr | 1,771 | 73,555 | 11.5 | 372 | **98%** | सेवन ("consumption") -> seven, twice |
| bn | 1,088 | 69,132 | 25.0 | 425 | **100%** | |
| pa | 2,101 | 95,575 | 33.5 | 694 | **99%** | ਫੋਲਿੰਗ ("folding") -> following |
| gu | 1,717 | 81,629 | 15.9 | 521 | **98%** | સેવન ("consumption") -> seven; વોટર ("voter") -> water |
| or | 1,636 | 79,995 | 19.4 | 424 | **100%** | |
| ur | 1,610 | 97,176 | 30.8 | 551 | **97%** | ہالی ووڈ (Hollywood) -> ہالی wood, twice; روم (Rome) -> room |
| ta | 2,449 | 153,319 | 8.2 | 373 | **100%** | |
| te | 2,463 | 69,452 | 35.2 | 770 | **99%** | ట్రయల్ కోర్టు (trial court) -> trail |
| kn | 1,785 | 112,250 | 14.3 | 486 | **100%** | |
| ml | 2,031 | 44,626 | 24.2 | 403 | **99%** | ഫെയ്സ് ബുക്ക് (Facebook in two words) -> face |

Every wrong rewrite is now blocklisted except ഫെയ്സ് (mostly a correct "face"). Known homographs of
the same kinds that were not sampled are blocklisted too: Urdu سیون ("seam"), Punjabi ਉਬਰ ("recover"),
Hindi व्हेन (glossed "van"), Kannada ಬಿಲ್ಲು ("bow"), ಸರ್ರ (onomatopoeia), ಕರ್ಕು, ಬಟ್ಟ, ಬಟ್ಟು. The
allowlist adds seven Kannada loanwords that Kathbath uses and IndicVoices glossed too rarely (ಕ್ಲಿಕ್
click, ಮಿಲಿಟರಿ military, ವೈರಲ್ viral, ನೋಟಿಸ್ notice, ಸ್ಪಿನ್ spin, ಅಪಾರ್ಟ್ಮೆಂಟ್ apartment, ಫ್ಲಾಟ್ flat).

Rates are per Kathbath word; Kathbath is news-style read text, so they are higher than in
conversational speech. Kannada was reviewed on 10,209 sentences (one local and four remote shards);
the other languages on 4,100-23,000.

**Examples** (native -> Latin, context in brackets; all correct unless marked):

- **hi:** पुलिस -> police (कोतवाली पुलिस ने); पासपोर्ट -> passport; कैमरा -> camera; मिल्क -> milk (the
  film "Milk"); ट्रिपल -> triple; *wrong:* उबर -> uber (दिखने से पहले ही उबर सके).
- **mr:** परफेक्ट -> perfect; रजिस्ट्रेशन -> registration; टेम्पो -> tempo; ट्रोल -> troll; बेस्ट -> best
  (BEST buses); *wrong:* सेवन -> seven (तंबाखू सेवन).
- **bn:** উইকেট -> wicket; সিকিউরিটি -> security; অ্যান্ড -> and (স্কুল অ্যান্ড কলেজ); ব্লিচিং -> bleaching;
  পয়েন্ট -> point (পয়েন্ট ব্ল্যাঙ্ক).
- **pa:** ਪੈਨਸ਼ਨ -> pension; ਵੈਟਰਨਰੀ -> veterinary; ਸਵੈਟਰ -> sweater; ਮਾਰਚ -> march (ਫਲੈਗ ਮਾਰਚ);
  *wrong:* ਫੋਲਿੰਗ -> following (ਸਕ੍ਰੀਨ ਫੋਲਿੰਗ ਟੇਬਲ).
- **gu:** હોમવર્ક -> homework; પેસેન્જર -> passenger; ટેરેસ -> terrace; ફ્રેક્ચર -> fracture; *wrong:*
  વોટર -> water (વોટર આઈડી, "voter ID").
- **or:** ଟ୍ରାଫିକ୍ -> traffic; ଇନ୍ସୁରାନ୍ସ -> insurance; କଷ୍ଟ୍ୟୁମ୍ -> costume; ଷ୍ଟାପ୍ -> staff; ହାଉସ -> house
  (Patiala House court).
- **ur:** ہیلتھ -> health; سکیورٹی -> security; ٹورنامنٹ -> tournament; بانڈ -> bond (the Bond film);
  *wrong:* روم -> room (روم میں ایک شخص پوپ, "Rome").
- **ta:** கலெக்டர் -> collector; மேரேஜ் -> marriage; கோர்ட்டு -> court; கஸ்டமைஸ் -> customise; வீலர் ->
  wheeler (டூ வீலர்).
- **te:** రెవెన్యూ -> revenue; ట్రైను -> train; సబ్సిడీ -> subsidy; థ్రిల్లర్ -> thriller; *wrong:*
  ట్రయల్ -> trail (ట్రయల్ కోర్టు).
- **kn:** ಸೇಲ್ಸ್ -> sales; ಕಾಲೇಜು -> college; ಸ್ಕ್ಯಾನಿಂಗ್ -> scanning; ಅಡಿಕ್ಟ್ -> addict; ಲೇಬರ್ -> labour
  (Labour party).
- **ml:** ഡെലിവറി -> delivery; ഡയാലിസിസ് -> dialysis; എക്സ്ട്രാ -> extra (എക്സ്ട്രാ കവർ); ന്യൂക്ലിയർ ->
  nuclear; *wrong:* ഫെയ്സ് -> face (ഫെയ്സ് ബുക്കില്).

**Reviewer caveat.** The 1,100 samples were judged by an automated reviewer (the LLM agent that built
the lexicon) from the sentence context; no human or native speaker checked them. For Odia, Gujarati,
Punjabi and Malayalam the judgements rest more on the English word being recognisable in the script
than on knowledge of the language. The precision of a 100-sample
review has a wide interval (roughly +-3 points near 97%).

## Limitations

- The rewrite is context-free: a native homograph of a lexicon entry is rewritten wherever it
  occurs. The review found such cases (Marathi सेवन "consumption" vs seven, Hindi उबर "recover" vs
  Uber, Punjabi ਫੋਲਿੰਗ glossed as "following" but used for "folding", Kannada ಬಿಲ್ಲು "bow" vs bill);
  they are blocklisted, but unsampled homographs may remain.
- The lexicon only knows words IndicVoices annotators glossed at least three times. Formal
  vocabulary in Kathbath and Shrutilipi (news, science) has many loanwords IndicVoices never
  glossed, which stay native; recall is not measured.
- Spelling follows the annotators' gloss, including British forms (customise, tyre, theatre) and
  Indian English (pincode, readymade).
- A suffixed loanword ("ಬ್ಯಾಂಕಿನಲ್ಲಿ") is never rewritten, so native-only sources still contain some
  native-script loanwords in inflected forms.
- The review judges whether the native word is the English word and the Latin spelling is right.
  It does not judge whether a fully nativised loan (पुलिस, फिल्म) *should* be Latin; the project
  convention, set by the IndicVoices glosses, says it should.
