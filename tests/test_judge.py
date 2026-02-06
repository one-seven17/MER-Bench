from judge import *


def test_judge_example():
    sample = MemeSample(
        sample_id="21.png",
        src_image_path="data/Original/21.png",
        gen_image_path="data/EditedResults/FLUX.2-klein-4B/21.png",
        src_emotion="angry",
        tgt_emotion="calm",
        src_caption_text="Me when the customer puts their money on the\ncounter instead of my outstretched hand\nMemeCenter.com\n",
        tgt_caption_text="Me when the customer smiles and thanks me while paying — what a great start to the day!",
        edit_instruction="Transform the expression into calm/content, warmer softer lighting, preserve layout and subject.",
    )

    client = JudgeClient(
        model="gpt-5.2",
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
