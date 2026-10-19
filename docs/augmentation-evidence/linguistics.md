# What can vary without changing a sign's meaning: linguistic grounding for ASL skeleton augmentation

Scope: isolated-sign recognition (WLASL-style clips), signer represented as a 3-D skeleton (SMPL-X body, MANO hands, MediaPipe landmarks). Research date: 2026-10-11.

## Verification labels

- **VERIFIED**: I read the primary source's abstract or full text, at the URL given.
- **VERIFIED-SECONDARY**: I read a peer-reviewed or authoritative source that describes the claim, but not the primary source itself. The citing source is named.
- **EXISTS**: Only the bibliographic record is confirmed (Crossref, DOI or publisher). The content claim is not verified.
- **UNVERIFIED**: The claim comes from search-engine summaries, practitioner pages or recall. Treat it as a lead, not as evidence.

URLs marked "(Crossref API)" were read as Crossref JSON records, including the abstract where Crossref holds one. The DOI link is given so the source can be resolved.

---

## 1. Phonological parameters and minimal pairs

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 1.1 | Stokoe's 1960 monograph analysed signs into tab (location), dez (handshape) and sig (movement). It was reprinted as a classic in JDSDE in 2005. | Stokoe, W. C. (1960/2005). *Sign Language Structure: An Outline of the Visual Communication Systems of the American Deaf.* Studies in Linguistics, Occasional Papers 8; reprinted in *J. Deaf Stud. Deaf Educ.* 10(1), 3–37. | https://doi.org/10.1093/deafed/eni001 (Crossref API); Wikipedia summary https://en.wikipedia.org/wiki/American_Sign_Language_phonology | EXISTS (reprint record). The tab/dez/sig mapping is VERIFIED-SECONDARY (Wikipedia). |
| 1.2 | The usual parameter set is handshape, movement, location, palm orientation and non-manual signals. Stokoe did not originally treat orientation as a separate prime. | Wikipedia, "American Sign Language phonology" (tertiary). | https://en.wikipedia.org/wiki/American_Sign_Language_phonology | VERIFIED-SECONDARY (tertiary) |
| 1.3 | Minimal pairs from Battison (1974), as tabulated on Wikipedia: location APPLE/ONION; handshape CAR/WHICH; movement PROPOSE/MAYBE; orientation TEMPORARY/TRAIN. MOTHER/FATHER is cited as a location contrast. | Battison (1974), via Wikipedia. | https://en.wikipedia.org/wiki/American_Sign_Language_phonology | VERIFIED-SECONDARY (tertiary) |
| 1.4 | Battison (1974) treats ASL as having a level analogous to phonology and grounds the analysis of variation in "articulatory dynamics" of the hands and body. | Battison, R. (1974). Phonological deletion in American Sign Language. *Sign Language Studies* 5, 1–19. | https://doi.org/10.1353/sls.1974.0005 (Crossref API abstract) | VERIFIED (abstract) |
| 1.5 | SUMMER/UGLY/DRY as a location minimal triple. Lifeprint (a practitioner site, not peer reviewed) puts formal DRY at lip level and UGLY just below the nose, usually with a negative facial expression. It says the two "sometimes may even look the same" in casual signing. SUMMER is not discussed on that page. I found no peer-reviewed source for the triple. | Vicars, B. Lifeprint ASL University, "DRY". | https://lifeprint.com/asl101/pages-signs/d/dry.htm | VERIFIED for what the page says. It is practitioner-grade, and the SUMMER part is UNVERIFIED. |
| 1.6 | Brentari's Prosodic Model holds that consonant-like and vowel-like units occur simultaneously and that movements are the basic prosodic units. | Brentari, D. (1998). *A Prosodic Model of Sign Language Phonology.* MIT Press. | https://doi.org/10.7551/mitpress/5644.001.0001 (Crossref API description) | VERIFIED (publisher description) |
| 1.7 | Sandler's work on sign phonology. | Sandler, W. & Lillo-Martin, D. (2006). *Sign Language and Linguistic Universals.* Cambridge UP. | https://doi.org/10.1017/cbo9781139163910 (Crossref API) | EXISTS only. I verified no content claim from Sandler. |
| 1.8 | Allophonic variation exists below the contrastive level. For example, finger bending and thumb position in the /B/ handshape vary with context, such as whether the fingertips touch the body. | Wikipedia, "American Sign Language phonology". | https://en.wikipedia.org/wiki/American_Sign_Language_phonology | VERIFIED-SECONDARY (tertiary) |

**How fine the contrasts are.** Location contrasts on the face can be only a few centimetres apart. The Lifeprint material on DRY and UGLY (1.5) and the gender pairs below (2.6) show this. Location contrasts are therefore fine-grained in absolute terms, even though location is perceived less categorically (§5).

## 2. Phonetic and sociolinguistic variation that preserves meaning

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 2.1 | Forehead signs such as KNOW are often produced lower. About 2,900 tokens from more than 200 signers at 7 US sites show that lowering is constrained by both linguistic and social factors, with evidence of change in progress. The secondary account in Mauk & Tyrone (2.3) says function words lower more often. (The project is often cited as 2001; the journal article is 2002.) | Lucas, C., Bayley, R., Rose, M. & Wulf, A. (2002). Location variation in American Sign Language. *Sign Language Studies* 2(4), 407–440. | https://doi.org/10.1353/sls.2002.0020 (Crossref API abstract) | VERIFIED (abstract) |
| 2.2 | Lowering is a form of phonetic reduction. Motion capture of WONDER from 6 native signers, in 2 contexts at 3 rates, showed lowering in **all** signers. Rate, phonetic context and position in the utterance all mattered. | Tyrone, M. E. & Mauk, C. E. (2010). Sign lowering and phonetic reduction in American Sign Language. *Journal of Phonetics* 38(2), 317–328. doi:10.1016/j.wocn.2010.02.003 | Europe PMC record PMC2895670 (abstract read via Europe PMC REST API): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC2895670 | VERIFIED (abstract) |
| 2.3 | Faster signing and lower neighbouring signs increase lowering, consistent with "general principles of coarticulation". Vertical location of KNOW tracked the *following* sign (r² ≈ 0.78) more than the preceding one (r² ≈ 0.14). In fast signing the preceding point's endpoint dropped by about 150 mm. Shifts also occurred left–right and front–back. | Mauk, C. E. & Tyrone, M. E. (2012). Location in ASL: Insights from phonetic variation. *Sign Language & Linguistics* 15(1), 128–146. | https://pmc.ncbi.nlm.nih.gov/articles/PMC4606891 | VERIFIED (full text) |
| 2.4 | With 4 signers, forehead signs were lower in low environments and lowered "more often and to a greater extent" at fast rates. Rate also caused unexpected leftward shifts toward the midline. Shifting forward or sideways does not cross a phonological boundary and "cannot create a different lexical item". Location is more flexible in production than models assume. | Tyrone, M. E. & Mauk, C. E. (2012). Phonetic reduction and variation in ASL: A quantitative study of sign lowering. *Laboratory Phonology* 3(2), 425–453. | https://pmc.ncbi.nlm.nih.gov/articles/PMC4489421 | VERIFIED (full text) |
| 2.5 | Corpus study: more than 3,000 tokens from 6 signers, for signs at the face, head and neck. The evidence fits undershoot mostly, with a categorical lowering process in a minority of signs. Signers "systematically avoid contacting their eyes". Lowering is large for the forehead, smaller for eye level, and small or arbitrary from nose level down. | Russell, K., Wilkinson, E. & Janzen, T. (2011). ASL sign lowering as undershoot: A corpus study. *Laboratory Phonology* 2, doi:10.1515/labphon.2011.015 | Author draft https://home.cc.umanitoba.ca/~krussll/papers/Russell,%20Wilkinson,%20and%20Janzen%202011%20draft.pdf (full text extracted and read) | VERIFIED (author's revised draft) |
| 2.6 | Gender pairs differ only in location: the male sign is at the forehead, the female sign at the mouth or chin. BROTHER (forehead) shows a downward tail of lowered tokens, while SISTER (chin) stays tightly clustered. MOTHER/FATHER and MAN/WOMAN show the same pattern. **Lowering is directional and acts on the higher member. Lowering the low member is not observed.** | Russell, Wilkinson & Janzen (2011), §3.3, Fig. 6. | same as 2.5 | VERIFIED |
| 2.7 | Rate-driven undershoot of handshapes and locations, with more undershoot at higher speed. | Mauk (2003, dissertation), as cited in Tyrone & Mauk (2012). | https://pmc.ncbi.nlm.nih.gov/articles/PMC4489421 | VERIFIED-SECONDARY |
| 2.8 | Location-to-location coarticulation in ASL is weaker than vowel-to-vowel coarticulation in speech. Perceivers, signers and non-signers alike, showed "relatively little evidence of sensitivity" to it, even at the closest distances. | Grosvald, M. & Corina, D. P. (2012). The perceptibility of long-distance coarticulation in speech and sign. *Sign Language & Linguistics* 15(1), 73–103. | https://doi.org/10.1075/sll.15.1.04gro (Crossref API abstract) | VERIFIED (abstract) |
| 2.9 | 1-handshape variation. More than 5,000 tokens from more than 200 signers show three main variants. Variation is conditioned by grammatical function (the first-order linguistic constraint), by the preceding and following segments (progressive and regressive assimilation), and by social factors. The pattern is similar across US regions. | Bayley, R., Lucas, C. & Rose, M. (2002). Phonological variation in ASL: The case of 1 handshape. *Language Variation and Change* 14, 19–53. | https://doi.org/10.1017/s0954394502141020 (Crossref API abstract) | VERIFIED (abstract). The variant forms themselves (e.g., thumb-extended "L-like" and open-hand variants) are UNVERIFIED because the abstract does not list them. |
| 2.10 | Register: whispering (short-distance signing) versus shouting (long-distance signing) in Sign Language of the Netherlands. Shouting enlarges movement and whispering reduces it. Parameters can also trade off: in enlarged signing, handshape or orientation changes may be carried out as location changes, and in reduced signing the reverse happens. Lexical entries are argued to specify *perceptual targets*; joint and arm states are filled in phonetically. | Crasborn, O. (2001). *Phonetic Implementation of Phonological Categories in Sign Language of the Netherlands.* PhD, Leiden (LOT 48). | Abstract on LINGUIST List 16.2765: https://dev.linguistlist.org/issues/16/2765/ | VERIFIED (dissertation abstract) |
| 2.11 | Stress does not have one primary cue. It shows up as sharp sign boundaries, signs produced **higher** in the signing space, added path movement for signs that lack one, repetition, and non-manual changes. Size and intensity changes were only partially confirmed. | Wilbur, R. B. & Schick, B. S. (1987). The effects of linguistic stress on ASL signs. *Language and Speech* 30(4), 301–323. | https://doi.org/10.1177/002383098703000402 (Crossref API abstract) | VERIFIED (abstract) |
| 2.12 | Proximalization and distalization (moving articulation to joints nearer to or farther from the torso) occur in adult learners. | Mirus, Rathmann & Meier (2001), *Signed Languages* volume, Gallaudet UP. | https://doi.org/10.2307/j.ctv2rh296t.10 (Crossref API) | EXISTS only |

## 3. Hand dominance

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 3.1 | "Signing with the left or the right hand is not lexically contrastive in ASL." Dominance reversal (Frishberg 1985) therefore does not produce homophony. The dominant and weak hands are abstract phonological roles, not physical hands. | Becker, A. A. (2022). The effect of iconicity on weak hand drop in ASL. *Proc. Annual Meetings on Phonology* 9. doi:10.3765/amp.v9i0.5305 | https://par.nsf.gov/biblio/10401592/media/xml | VERIFIED (full text). Frishberg (1985) is VERIFIED-SECONDARY. |
| 3.2 | Corpus NGT, 92 signers: the dominant hand in a given sign does not necessarily match the signer's preferred hand. Reversal rates vary a lot between signers. Most reversals seem to be functional, typically marking contrast at some level of the grammar, though some have no communicative reason. | EU FP7 project "On the other hand" (Radboud University), CORDIS report. | https://cordis.europa.eu/project/id/210373/reporting/pl | VERIFIED (project report; not peer reviewed) |
| 3.3 | Left- and right-handed signers all responded faster to right-handed models for simple one-handed signs, which the authors attribute to frequency. Left-handers preferred input matching their own handedness for complex two-handed asymmetrical signs. | Watkins, F. & Thompson, R. (2017). *Cognition* 164, 144–149. doi:10.1016/j.cognition.2017.03.019 | https://research.birmingham.ac.uk/en/publications/the-relationship-between-sign-production-and-sign-comprehension-w/ | VERIFIED (abstract) |
| 3.4 | Engineering evidence. In ChicagoFSWild, 522 of 7,425 fingerspelling sequences are left-handed (about 7%). Horizontally flipping the left-handed videos and training a right-hand-only recognizer gave the same accuracy (76.0% versus 76.1% letter accuracy). | Shi, B. (2023). *Toward ASL Processing in the Real World* (PhD thesis, TTIC), arXiv:2308.12419. | https://arxiv.org/pdf/2308.12419 | VERIFIED (full text) |
| 3.5 | Battison's **Symmetry Condition**: if both hands move, they must share handshape and movement (symmetric or alternating). **Dominance Condition**: if the handshapes differ, the weak hand is static and limited to a small unmarked set. Napoli & Wu show the weak hand is unmarked even when the hands move as a unit. They identify four symmetry transformations (reflection, rotation, translation, glide reflection) and suggest these may follow from physiology. | Battison, R. (1978). *Lexical Borrowing in ASL.* Linstok; Napoli, D. J. & Wu, J. (2003). *Sign Language & Linguistics* 6(2), 123–205. | https://doi.org/10.1075/sll.6.2.03nap (Crossref API abstract) | VERIFIED (Napoli & Wu abstract). Battison 1978 is VERIFIED-SECONDARY. |
| 3.6 | In Battison's type-3 signs the weak hand is limited to seven "unmarked" handshapes. | Becker (2022), citing Battison (1978). | https://par.nsf.gov/biblio/10401592/media/xml | VERIFIED-SECONDARY |
| 3.7 | Handedness as a formal phonological parameter, covering weak drop, weak add and weak prop. | Lepp, De Sisto & Shterionov (2026). LREC 2026 sign-lang workshop, pp. 275–286. doi:10.63317/3per9mf2462d | https://www.sign-lang.uni-hamburg.de/lrec/pub/26032.html | UNVERIFIED (search summary only; the page was not opened) |

## 4. Weak drop (weak-hand deletion)

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 4.1 | Padden & Perlmutter name the process "Weak Drop" and place it in a **post-lexical** phonological component. Its outputs must be blocked from the derivational Characteristic Adjective Rule. | Padden, C. & Perlmutter, D. (1987). ASL and the architecture of phonological theory. *Natural Language & Linguistic Theory* 5, 335–375. doi:10.1007/bf00134553 | https://cnlse.es/es/recursos/biblioteca/american-sign-language-and-architecture-phonological-theory | VERIFIED (abstract) |
| 4.2 | Weak hand drop is a "post-lexical phonological process" in which only the dominant hand of a two-handed sign surfaces. Battison (1974) noted that it could neutralize minimal pairs such as PURPLE/PARTY. | Becker (2022) | https://par.nsf.gov/biblio/10401592/media/xml | VERIFIED. The Battison detail is VERIFIED-SECONDARY. |
| 4.3 | **Blocking conditions** (Brentari 1998, as tabulated in Becker 2022): alternating movement with the same handshape (BICYCLE); continuous contact between the hands (WITH); contact between analogous hand parts in type-2 signs (SCHOOL); certain dominant-hand orientations (WHEN, JAIL); dominant-hand contact with a weak-hand part other than the palm in type-3 signs (AREA). | Brentari (1998), via Becker (2022). | https://par.nsf.gov/biblio/10401592/media/xml | VERIFIED-SECONDARY |
| 4.4 | Battison (1974): weak drop is more acceptable when the weak hand iconically represents a surface. Type-2 signs were judged bad in isolation but sometimes appeared one-handed in spontaneous signing. Van der Kooij (2001), for Sign Language of the Netherlands, found alternating signs rejected in 16 of 36 cases, with the semantic or iconic motivation of two-handedness mattering more than form. | Via Becker (2022). | https://par.nsf.gov/biblio/10401592/media/xml | VERIFIED-SECONDARY |
| 4.5 | Forty Deaf signers rated one-handed forms of highly iconic two-handed signs *more* acceptable than one-handed forms of similar, less iconic signs, so iconicity affects weak drop. | Becker (2022) | https://par.nsf.gov/biblio/10401592 | VERIFIED (abstract) |

## 5. Categorical perception

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 5.1 | Using computer-generated continua, Deaf native signers showed categorical perception **only for phonologically contrastive handshape** continua. "Neither group exhibited CP for the place of articulation continua." Both groups could still *identify* place-of-articulation categories. | Emmorey, K., McCullough, S. & Brentari, D. (2003). Categorical perception in ASL. *Language and Cognitive Processes* 18(1), 21–45. doi:10.1080/01690960143000416 | Description in Baker, Idsardi, Golinkoff & Petitto (2005), *Memory & Cognition* 33(5), 887–904: https://terpconnect.umd.edu/~idsardi/papers/2005memcog.pdf (full text read) | VERIFIED-SECONDARY. The primary abstract was unavailable (403). |
| 5.2 | Baker et al. (2005) replicated categorical perception for contrastive handshapes with natural stimuli. They studied only handshape because of "a lack of CP effects" for other contrasts such as place of articulation. | Baker et al. (2005) | same URL as 5.1 | VERIFIED |
| 5.3 | Language experience (native versus non-native) affected discrimination of **handshape** near category prototypes. **Place-of-articulation** stimuli showed a category-structure effect that did *not* interact with language background. | Morford, J. P., Grieve-Smith, A. B., MacFarlane, J., Staley, J. & Waters, G. (2008). Effects of language experience on the perception of ASL. *Cognition*. | Europe PMC abstract (PMC2639215): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC2639215 | VERIFIED (abstract) |
| 5.4 | Deaf signers show categorical perception for **linguistic** facial expressions but not for affective ones. Hearing non-signers show it for both. | McCullough, S. & Emmorey, K. (2009). *Cognition* 110, 208–221. doi:10.1016/j.cognition.2008.11.007 | Europe PMC abstract (PMC2681241): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC2681241 | VERIFIED (abstract) |

Implication: handshape is perceived categorically, so jitter within a handshape category is tolerated perceptually, but crossing a category boundary changes the sign. Location is perceived more continuously, which is consistent with the broad lowering in §2. However, location minimal pairs exist and some are only a few centimetres apart (BROTHER/SISTER, DRY/UGLY), so location tolerance is direction-dependent rather than wide.

## 6. Timing, duration and aspect

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 6.1 | Signs take longer to produce than spoken words, but propositions take about the same time in both modalities. | Bellugi, U. & Fischer, S. (1972). A comparison of sign language and spoken language. *Cognition* 1, 173–200. doi:10.1016/0010-0277(72)90018-2 | Crossref record only | EXISTS. The claim is UNVERIFIED (search summaries only). |
| 6.2 | Signers change rate by changing **sign duration**, not the number of signs. Faster rate also shortens pauses, brow raises, lowered brows and blinks. The paper discusses the difference between grammatical and non-grammatical nonmanuals. | Wilbur, R. B. (2009). Effects of varying rate of signing on ASL manual signs and nonmanual markers. *Language and Speech* 52, 245–285. | https://doi.org/10.1177/0023830909103174 (Crossref API abstract) | VERIFIED (abstract) |
| 6.3 | Telic verbs (THROW, HIT) mark their endpoint with significantly greater **deceleration** than atelic verbs (SWIM, TRAVEL). The marker is robust to sentence position, while duration and peak speed are affected by phrase-final lengthening. | Malaia, E. & Wilbur, R. B. (2011/2012). Kinematic signatures of telic and atelic events in ASL predicates. *Language and Speech* 55(3), 407–421. | https://doi.org/10.1177/0023830911422201 (Crossref API abstract) | VERIFIED (abstract) |
| 6.4 | Klima & Bellugi (1979) catalogue aspectual modulations that change tension, reduplication, pauses and **rate of movement**. They list 15 aspect types. Rathmann (2005) reduces these to continuative (movement extended in time), iterative (reduplication), habitual, hold and conative. Continuative is often realized as "slow reduplication". | Klima, E. & Bellugi, U. (1979). *The Signs of Language.* Harvard UP; via van Boven, C. (2018). THROUGH as a free aspect marker in NGT. *Linguistics in Amsterdam* 11(1), 1–21. | https://linguisticsinamsterdam.nl/article/download/27873/28344/77142 (full text extracted and read) | VERIFIED-SECONDARY |
| 6.5 | Specific Klima & Bellugi glosses (protractive = held, no repetition; incessant = short tense repeated movement; durational = circular reduplication; habitual = rapid non-tense repetition; continuative = elliptical). | Klima & Bellugi (1979) | none opened | UNVERIFIED (search summaries) |
| 6.6 | ASL-LEX 2.0 sign durations (onset to offset) have a mean of 851 ms (SD 301). They come from a single signer's model productions, and the authors advise caution because of phonetic and individual variation. | Sehyr et al. (2021), see §8. | https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685 | VERIFIED |

## 7. Non-manual markers

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 7.1 | Brow raise ('br') marks [−wh] operator structures such as yes/no questions, topics and conditionals. ASL has three distinct brow positions (raised, furrowed, neutral) tied to [−wh], [+wh] and no operator. | Wilbur, R. B. & Patschke, C. (1999). Syntactic correlates of brow raise in ASL. *Sign Language & Linguistics* 2(1), 3–41. | https://doi.org/10.1075/sll.2.1.03wil (Crossref API abstract) | VERIFIED (abstract) |
| 7.2 | The negative headshake is used syntactically to mark negation and aligns with syntactic constituents. It may accompany a negative sign, but "no negative sign was found to require a headshake". | Veinberg, S. & Wilbur, R. B. (1990). *Sign Language Studies* 68, 217–244. | https://doi.org/10.1353/sls.1990.0013 (Crossref API abstract) | VERIFIED (abstract) |
| 7.3 | Some signs carry a lexically required mouth morpheme. NOT-YET requires TH, while LATE has the same manual form and no mouth morpheme. | Duarte, A. et al. (2021). How2Sign. CVPR, arXiv:2008.08143. | https://arxiv.org/pdf/2008.08143 (full text read) | VERIFIED-SECONDARY (dataset paper, not linguistics) |
| 7.4 | Adverbial mouth morphemes such as MM ('normal, relaxed') and TH ('careless'). CHA ('large') is attributed to Liddell (1980) and to Bickford & Fraychineaud. | Liddell, S. (1980). *American Sign Language Syntax.* Mouton (doi:10.1515/9783112418260). Bickford & Fraychineaud, "Mouth morphemes in ASL: A closer look". | Liddell record via Crossref; Bickford PDF returned 404 | Liddell EXISTS. The specific glosses are UNVERIFIED. |
| 7.5 | Linguistic facial expressions are perceived categorically by Deaf signers (see 5.4). | McCullough & Emmorey (2009) | see 5.4 | VERIFIED |
| 7.6 | Baker-Shenk (1983) on facial behaviour. Mouthing literature: Nadolske & Rosenstock (2007), mouthings in ASL, *Visible Variation*, doi:10.1515/9783110198850.35. | — | Crossref | EXISTS only |

For isolated WLASL-style citation signs, sentence-level nonmanuals (brow raise, headshake) are not part of the lexical item. Lexically specified mouth patterns (NOT-YET with TH) and the "negative facial expression" of UGLY (1.5) are part of the lexical item.

## 8. ASL-LEX 2.0

| # | Claim | Citation | URL | Status |
|---|---|---|---|---|
| 8.1 | ASL-LEX 2.0 covers 2,723 sign entries (2,663 lemmas). Phonological coding follows Brentari's (1998) Prosodic Model and adds nine new features over ASL-LEX 1.0, which had 993 signs. | Sehyr, Z. S., Caselli, N., Cohen-Goldberg, A. M. & Emmorey, K. (2021). The ASL-LEX 2.0 Project. *J. Deaf Stud. Deaf Educ.* 26(2), 263–277. doi:10.1093/deafed/enaa038. ASL-LEX 1.0: Caselli et al. (2017), *Behavior Research Methods* 49, 784–801, doi:10.3758/s13428-016-0742-0 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685 | VERIFIED |
| 8.2 | Per-sign phonological fields: **Sign Type**, following Battison (one-handed; two-handed symmetrical or alternating; asymmetrical with same handshape; asymmetrical with different handshape). **Major Location** (head, arm, body, non-dominant hand, neutral). **Minor Location** (8 per major location, plus "away"). **Second Minor Location**. **Contact** (±). **Selected Fingers**. **Flexion**. **Spread**. **Thumb Position** (open/closed). **Thumb Contact**. **Handshape** (curated categories). **Non-dominant Handshape**, which includes the values SymmetryViolation, DominanceViolation and Lax. **Path Movement** (arc, circular, straight, z, x, none, other). **Internal Movement**: flexion change, spread change and ulnar rotation (the paper's term for wrist twist). **Repeated Movement** (±). | Sehyr et al. (2021) | same | VERIFIED |
| 8.3 | Other per-sign fields: frequency ratings from Deaf signers; iconicity ratings from Deaf signers and from hearing non-signers; transparency; neighbourhood density and complexity; lexical class; initialized, fingerspelled or compound status; sign and video duration. Data are on OSF, with an interactive site at asl-lex.org. | Sehyr et al. (2021); search summary | same | Mostly VERIFIED. The "initialized/fingerspelled/compound" flag comes from a search summary of the same paper and is UNVERIFIED in my own reading. |
| 8.4 | **License, which needs attention:** the asl-lex.org page lists the database and visualization as **CC BY-NC 4.0** and marks the site and videos "© ASL-LEX.org". The OSF project (zpha4) metadata says **CC BY 4.0**. The JDSDE article is CC BY-NC-ND 4.0. The conservative reading is to assume non-commercial use, attribute, and cite both papers. | asl-lex.org; OSF API | https://asl-lex.org/ ; https://api.osf.io/v2/nodes/zpha4/?embed=license (OSF project https://osf.io/zpha4/) | VERIFIED (the two sources disagree) |

**Gating augmentations per sign with ASL-LEX fields.** These are design suggestions derived from the claims above, not published findings:

- `Sign Type == one-handed`: weak-hand perturbation is irrelevant. Mirroring to the other hand is safe (3.1).
- `Sign Type ∈ {symmetrical/alternating}` with `Contact == −`: weak drop is plausible. Block it if `Repeated Movement` together with alternating movement suggests a BICYCLE-type sign (4.3).
- `Sign Type ∈ {asymmetrical-same, asymmetrical-different}` or `Major Location == non-dominant hand`: **never** apply weak drop. Keep the relative pose between the hands fixed (4.3).
- `Major Location == head` with a forehead `Minor Location`: lowering is allowed only downward, partial (forehead → eye/cheek band), never into the eyes, and not when a lower-location minimal pair exists (2.5, 2.6).
- `Major Location == neutral`: location translation and scaling are broadly safe (2.3, 2.10).
- `Selected Fingers`, `Flexion`, `Spread`, `Thumb Position`: hand jitter must keep each feature inside its category, and the thumb is especially sensitive (5.1). Large MANO PCA noise that flips flexion or thumb class is forbidden.
- `Repeated Movement == +` or `Path Movement == circular`: avoid adding or removing cycles, and avoid non-uniform warps that make movement look elliptical, held or tense (6.4).
- `Internal Movement` (flexion change, ulnar rotation): time warps must preserve the start and end handshape and orientation. Do not crop them.

---

## Implications for augmentation

### Safe (meaning-preserving, with literature support)

1. **Left/right mirroring of the whole skeleton**, swapping the MANO hand assignments and SMPL-X left/right joints. Dominance is not lexically contrastive (3.1). Left-handers are about 7–10% of signers (3.4). Mirroring is a geometric reflection, so non-manuals and torso move with it. Caveat: a few signs locate on a lateralized body part, such as the heart side; I found no source on this.
2. **Global uniform tempo scaling** within natural rate ranges. Rate is realized mainly as sign duration (6.2). Keep the scaling modest. The literature gives no numeric bound; use the spread of ASL-LEX and WLASL durations empirically (6.6).
3. **Small location shifts in neutral space, and front/back or left/right shifts of body-anchored signs**, which do "not cross a phonological boundary" (2.4). Rate-related shifts toward the midline are attested (2.4).
4. **Movement amplitude scaling (whisper ↔ shout)** within moderate limits (2.10, 2.11). Natural enlargement can turn handshape or orientation change into location change, so the converse (scaling the path) is natural.
5. **Within-category handshape jitter** (finger bend, spread, thumb within its class). Allophonic variation exists (1.8), and perception is categorical across boundaries but tolerant within them (5.1–5.3).
6. **Signer-shape and body-proportion variation** (SMPL-X betas) with joint targets re-anchored to body landmarks. This follows from the perceptual-target view in Crasborn (2.10). It is a design inference, not a published test.

### Conditional (allowed only under sign-specific gates)

1. **Forehead lowering**: only for head or forehead signs, only downward, partial, avoiding the eyes, and blocked when a lower minimal pair exists (BROTHER/SISTER, FATHER/MOTHER, MAN/WOMAN) (2.1–2.6). Lowering in natural data is graded and asymmetric. Prefer sampling from the observed lowering distribution over a fixed offset.
2. **Weak drop** (delete the weak hand): only for symmetrical or alternating signs without hand-to-hand contact, not BICYCLE-type alternating signs, never type-3 asymmetric signs or signs located on the weak hand (4.1–4.4). Iconicity modulates acceptability (4.5).
3. **Coarticulatory start and end shifts**: shift the onset and offset toward a plausible neighbouring sign's location. Isolated clips in WLASL may include transitions. Coarticulatory effects are weaker in sign than in speech (2.8) and are dominated by the *following* sign (2.3).
4. **Non-uniform time warping**: only if it preserves the endpoint deceleration profile (the telicity cue, 6.3), the number of repetitions, and holds.
5. **Facial and non-manual perturbation**: perturbing or neutralising brow and head channels is fine for citation forms, but lexically specified mouth patterns (NOT-YET/TH, 7.3) and lexical facial expressions (UGLY, 1.5) must be preserved.
6. **1-handshape variants** (thumb out, etc.) are conditioned by grammatical function and context (2.9). Treat them as a plausible variant only for index-handshape signs and keep the selected fingers fixed.

### Forbidden (likely to change lexical identity or morphology)

1. Crossing a **handshape category boundary**: changing selected fingers, flexion class, spread class or thumb class (5.1; 1.3 CAR/WHICH).
2. Moving a sign to a **contrastive location** (1.3 APPLE/ONION, MOTHER/FATHER; 2.6). This includes large vertical shifts on the face and moving chin or mouth signs *up* to the forehead.
3. Rotating the **palm orientation** across a category boundary (1.3 TEMPORARY/TRAIN).
4. Changing **movement shape or class**: straight versus arc versus circle, or adding or removing path movement (1.3 PROPOSE/MAYBE).
5. **Adding, removing or retiming repetitions**, or holding a sign, in ways that mimic aspectual modulations (iterative, continuative, habitual, protractive) (6.4). Do not loop-pad or duplicate cycles of verb signs.
6. **Weak drop** on contact, alternating, type-3 or weak-hand-location signs (4.3).
7. Mirroring **only one hand**, or swapping hand roles inside a two-handed asymmetric sign (breaks the Dominance Condition, 3.5).
8. Removing **lexically required mouth or face components** (7.3).

---

## Gaps and caveats

- I could not read the primary text of Emmorey et al. (2003) (403). The location result comes from Baker et al. (2005).
- Klima & Bellugi (1979) and Bellugi & Fischer (1972) are not verified at the primary level.
- No peer-reviewed source confirmed SUMMER/UGLY/DRY as a minimal triple. Use BROTHER/SISTER (Russell et al.) and the Battison pairs instead.
- I found no numeric perceptual tolerances (mm or degrees) for handshape or location. All bounds must be set empirically.
- The ASL-LEX licence differs between asl-lex.org (CC BY-NC 4.0) and OSF (CC BY 4.0).
- All studies except the How2Sign and ChicagoFSWild items are linguistic. None directly tests recognition-model robustness to these transformations.
