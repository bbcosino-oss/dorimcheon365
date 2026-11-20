"""도림천 365: AI 생태·걷기 관찰일지 (YOLO11 + Streamlit)."""
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="도림천 365", page_icon="🌿", layout="wide")

# YOLO COCO 클래스 분류: 종 식별이나 실제 개체군 조사에 사용할 수 없음.
GROUPS = {
    "person": ("보행·이용", "사람"),
    "bicycle": ("보행·이용", "자전거"),
    "bird": ("생태", "새"), "dog": ("생태", "개"), "cat": ("생태", "고양이"),
    "car": ("교통", "승용차"), "bus": ("교통", "버스"),
    "truck": ("교통", "트럭"), "motorcycle": ("교통", "오토바이"),
}
COLORS = {"생태": "#168568", "보행·이용": "#3478c0", "교통": "#dd8936", "기타": "#777777"}
COLUMNS = ["관찰일", "구간", "파일명", "그룹", "객체", "개수", "관찰날씨", "계절메모", "생태메모", "걷기메모"]

@st.cache_resource(show_spinner="최초 실행: YOLO11 경량 모델을 준비합니다.")
def load_model():
    return YOLO("yolo11n.pt")


def get_image(file):
    im = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    im.thumbnail((1280, 1280))
    return im


def infer(image, threshold):
    result = load_model().predict(image, conf=threshold, imgsz=640, verbose=False, device="cpu")[0]
    items = []
    for box in result.boxes:
        english = result.names[int(box.cls[0].item())]
        group, label = GROUPS.get(english, ("기타", english))
        items.append({"그룹": group, "객체": label, "원어": english,
                      "신뢰도": round(float(box.conf[0].item()), 3),
                      "좌표": tuple(map(int, box.xyxy[0].tolist()))})
    return items


def annotate(image, items):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for it in items:
        x1, y1, x2, y2 = it["좌표"]
        color = COLORS[it["그룹"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f'{it["원어"]} {it["신뢰도"]:.0%}'
        bb = draw.textbbox((0, 0), label)
        w, h = bb[2] - bb[0] + 10, bb[3] - bb[1] + 8
        y = max(0, y1-h)
        draw.rectangle((x1, y, x1+w, y+h), fill=color)
        draw.text((x1+5, y+3), label, fill="white")
    return out


def image_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def make_rows(day, zone, filename, items, weather, season, ecology, walking):
    counts = pd.DataFrame(items)
    if counts.empty:
        pairs = [("관찰", "탐지 없음", 0)]
    else:
        counts = counts.groupby(["그룹", "객체"], sort=True).size().reset_index(name="개수")
        pairs = counts[["그룹", "객체", "개수"]].itertuples(index=False, name=None)
    return [{"관찰일": day.isoformat(), "구간": zone, "파일명": filename,
             "그룹": group, "객체": name, "개수": n, "관찰날씨": weather,
             "계절메모": season, "생태메모": ecology, "걷기메모": walking}
            for group, name, n in pairs]


st.title("🌿 도림천 365")
st.markdown("**매일 걷고, 사진 찍고, AI와 함께 기록하는 도시의 사계**")
st.caption("YOLO11 객체 탐지 · Streamlit · Pandas | 사진 → 탐지 → 통계 → 생태·산책 기록 → CSV 보관")
st.info("AI가 새·사람·자전거·차량 등을 찾아줍니다. 실제 종 식별, 수질 평가, 정확한 개체수 측정은 하지 않습니다. 기록은 CSV를 내려받아 직접 보관하세요.")

with st.sidebar:
    st.header("📍 오늘의 관찰")
    day = st.date_input("관찰 날짜", value=date.today())
    zone = st.text_input("관찰 구간", value="도림천 산책로")
    weather = st.selectbox("날씨", ["맑음", "구름", "흐림", "비", "눈", "기타"])
    threshold = st.slider("객체 탐지 신뢰도 기준", 0.10, 0.80, 0.25, 0.05)
    alert_count = st.number_input("보행 이용 인원 알림 기준", min_value=1, max_value=100, value=3)
    st.caption("최초 실행 시 모델 가중치 약 5~6MB를 내려받습니다. 인터넷이 필요합니다.")

st.subheader("1. 사진과 관찰 기록")
photos = st.file_uploader("도림천 사진을 여러 장 선택하세요 (JPG·PNG)",
                          type=["jpg", "jpeg", "png"], accept_multiple_files=True)
with st.expander("📱 휴대폰 HEIC 사진 안내"):
    st.write("갤럭시 사진이 HEIC라면 휴대폰 갤러리에서 JPG로 저장하거나 변환 후 올려주세요. 데모 폴더의 사진은 JPG로 준비되어 있습니다.")
season = st.text_input("계절·식물 메모", placeholder="예: 버드나무 잎이 노랗게 물들었다")
ecology = st.text_area("생태 메모 (직접 관찰)", placeholder="예: 흰색 백로류 한 마리와 청둥오리 수컷 한 마리를 보았다")
walking = st.text_area("걷는 풍경·산책 에세이", placeholder="예: 아침 햇살이 하천 수면 위로 길게 비쳤다")

st.subheader("2. AI 탐지 결과")
if not photos:
    st.warning("분석을 시작하려면 위에서 사진을 올려주세요. ZIP에 포함된 demo_images 폴더의 JPG 사진을 사용해도 됩니다.")
    st.stop()

rows = []
all_objects = []
for i, photo in enumerate(photos, 1):
    st.markdown(f"#### {i}. {photo.name}")
    try:
        img = get_image(photo)
        with st.spinner(f"{photo.name} 분석 중..."):
            found = infer(img, threshold)
    except Exception as exc:
        st.error(f"{photo.name} 분석 오류: {exc}")
        continue
    result_img = annotate(img, found)
    left, right = st.columns(2)
    with left:
        st.image(img, caption="원본", use_container_width=True)
    with right:
        st.image(result_img, caption="YOLO 객체 탐지", use_container_width=True)
        st.download_button("탐지 이미지 PNG", image_png(result_img),
                           file_name=f"detected_{Path(photo.name).stem}.png", mime="image/png", key=f"im{i}")
    rows.extend(make_rows(day, zone, photo.name, found, weather, season, ecology, walking))
    all_objects.extend(found)

if not rows:
    st.error("분석할 수 있는 사진이 없습니다. 이미지 형식을 확인해 주세요.")
    st.stop()

st.subheader("3. 오늘의 생태·보행 통계")
today_df = pd.DataFrame(rows, columns=COLUMNS)
by_item = today_df.groupby(["그룹", "객체"], as_index=False)["개수"].sum()
count_groups = by_item.groupby("그룹")["개수"].sum().to_dict()
a, b, c, d = st.columns(4)
a.metric("분석 사진", len(set(today_df["파일명"])))
b.metric("생태 탐지", int(count_groups.get("생태", 0)))
c.metric("보행·이용 탐지", int(count_groups.get("보행·이용", 0)))
d.metric("교통 탐지", int(count_groups.get("교통", 0)))
if count_groups.get("보행·이용", 0) >= alert_count:
    st.warning("보행 관찰 알림: 설정한 탐지 기준 이상입니다. 여러 사진에 동일한 사람이 중복 탐지됐을 수 있습니다.")
if count_groups.get("생태", 0) > 0:
    st.success("생태 객체가 탐지됐습니다. 사진 속 새의 실제 종은 관찰 메모로 직접 기록해 주세요.")
if not all_objects:
    st.info("사진에서 객체를 찾지 못했습니다. 더 가까이 촬영한 사진이나 낮은 탐지 신뢰도를 시도하세요.")
x, y = st.columns(2)
with x:
    st.dataframe(by_item, hide_index=True, use_container_width=True)
with y:
    if not by_item.empty:
        st.bar_chart(by_item.set_index("객체")["개수"])

st.subheader("4. 장기 관찰 일지로 이어가기")
st.write("어제까지 다운로드한 CSV가 있다면 올려 오늘 기록과 합칠 수 있습니다. 클라우드 서버에 개인 사진이나 기록을 영구 저장하지 않습니다.")
history_file = st.file_uploader("기존 관찰일지 CSV 가져오기 (선택)", type=["csv"], key="history")
combined = today_df.copy()
if history_file is not None:
    try:
        history = pd.read_csv(history_file, encoding="utf-8-sig")
        if not set(COLUMNS).issubset(history.columns):
            st.error("기존 CSV의 열 구성이 다릅니다. 이 앱에서 내려받은 관찰일지 CSV를 사용하세요.")
        else:
            combined = pd.concat([history[COLUMNS], today_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=COLUMNS, keep="last")
            combined["개수"] = pd.to_numeric(combined["개수"], errors="coerce").fillna(0).astype(int)
            st.success(f"기존 기록과 합쳤습니다. 누적 {combined['관찰일'].nunique()}일, {len(combined)}행")
    except Exception as exc:
        st.error(f"CSV 파일을 읽지 못했습니다: {exc}")

st.dataframe(combined.sort_values(["관찰일", "파일명"], ascending=[False, True]),
             hide_index=True, use_container_width=True)
if combined["관찰일"].nunique() > 1:
    daily = combined.groupby(["관찰일", "그룹"], as_index=False)["개수"].sum()
    daily = daily.pivot(index="관찰일", columns="그룹", values="개수").fillna(0)
    st.markdown("#### 날짜별 관찰 추이 (사진별 탐지 합계)")
    st.line_chart(daily)

st.download_button("📥 도림천 365 관찰일지 CSV 다운로드", csv_bytes(combined),
                   file_name=f"dorimcheon365_{day.isoformat()}.csv", mime="text/csv", type="primary")
st.caption("주의: 사진이 많을수록 탐지 수가 커집니다. 결과는 실제 물새 개체수나 하천 생태 건강도와 다를 수 있으며, 같은 대상이 여러 사진에 반복될 수 있습니다.")
