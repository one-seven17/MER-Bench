from judge import *


def test_judge_example():
    sample = MemeSample(
        sample_id="2928.png",
        src_image_path="data/Original/2928.png",
        gen_image_path="data/EditedResults/Qwen-Image-Edit-2509/2928.png",
        src_emotion="happy",
        tgt_emotion="sad",
        src_caption_text='Me when I see a "me and the boys" meme, being a\nsocially awkward loner with no friends\n',
        tgt_caption_text='"Me when I see a \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"me and the boys\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" meme, feeling happy and content in my own company, knowing I have amazing friends for when I need them."',
        edit_instruction='"The character should have a gentle, warm smile instead of the tired, somber expression. The eyes should be open and slightly squinted with joy. The posture should be loose and comfortable, leaning slightly forward with ease. The lighting should be slightly warmer and brighter to convey a joyful and content vibe."',
    )

    client = JudgeClient(
        model="gemini-3-pro-preview",
        retry=RetryConfig(
            max_retries=6,
            base_delay_s=0.6,
            max_delay_s=20.0,
            jitter=0.25,
            timeout_s=120,
        ),
        max_in_flight=8,
        cache=CacheConfig(
            enabled=True, db_path="db.sqlite3", return_cached_errors=False
        ),
    )

    cfg = JudgeConfig(
        rationale_first=True,
        images_first=True,
        temperature=0.0,
        max_output_tokens=900,
        cost=CostConfig(enabled=True),
    )

    pipeline = JudgePipeline(
        client=client,
        cfg=cfg,
        ccfg=ConcurrencyConfig(sample_workers=2),
        rcfg=RunConfig(use_cache=True, force_refresh=True),
    )

    result = pipeline.evaluate_sample(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
