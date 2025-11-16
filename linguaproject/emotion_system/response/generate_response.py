'''
KoGPT 기반 상담사 응답 생성
styles.py에 입력해 상담사별 스타일을 적용 가능합니다(counselor_A, counselor_B 등)
'''

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def generate_response(emotion_label, user_text):
    style_map = {
        "불만": "공감하며 사과하고 해결 방안을 안내하는 말투로",
        "분노": "책임감 있게 사과하고 신속한 조치를 약속하는 말투로",
        "짜증": "불편을 인정하고 빠른 해결을 약속하는 말투로",
        "실망": "기대에 못 미친 점을 사과하고 개선을 약속하는 말투로",
        "불안": "안심시키고 절차를 명확히 설명하는 말투로",
        "혼란": "상황을 정리하고 명확하게 설명하는 말투로",
        "중립": "정중하고 간결하게 안내하는 말투로",
        "요청": "요청 사항을 확인하고 처리 절차를 안내하는 말투로",
        "호기심": "정보를 친절하게 설명하고 추가 안내를 제공하는 말투로",
        "감사": "감사 인사를 공손하게 전달하는 말투로",
        "기쁨": "긍정적인 분위기를 유지하며 감사 인사를 전하는 말투로",
        "감동": "진심 어린 감사와 응원의 말을 전하는 말투로",
        "슬픔": "공감하며 위로하고 필요한 도움을 안내하는 말투로",
        "피로": "간결하고 배려 있는 말투로 핵심만 안내하는 말투로"
    }
    style = style_map.get(emotion_label, "정중하고 간결한 말투로")
    prompt = f"""민원 상담 응답 생성기
사용자 감정: {emotion_label}
사용자 발화: {user_text}
상담사 응답 스타일: {style}
상담사 응답:"""
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=100, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)