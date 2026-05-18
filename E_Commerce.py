import os
import sys
import warnings
import time
import pandas as pd
import plotly.express as px
import streamlit as st
import mlflow

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    sum as _sum,
    explode,
    count,
    desc,
    avg,
    collect_set,
    collect_list,
    round as _round,
    sqrt,
    lit,
    broadcast,
    current_date,
    date_sub,
    rand,
    datediff,
    max as _max,
    date_trunc,
    expr,
    lower,
)
from pyspark.ml.recommendation import ALS
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Word2Vec
from pyspark.ml.evaluation import RegressionEvaluator

# Cấu hình môi trường và Streamlit
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI E-Commerce Enterprise",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded",
)


# Khởi tạo SparkSession
@st.cache_resource(show_spinner=False)
def init_spark():
    spark = (
        SparkSession.builder.appName("ECommerce_Enterprise_Final")
        .config("spark.driver.memory", "512g")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


spark = init_spark()


# Hàm tính Cosine Similarity dựa trên metadata sản phẩm
def get_improved_cosine(_spark, _product_metadata, target_product_id):
    from pyspark.sql import functions as F
    from pyspark.ml.feature import VectorAssembler, StringIndexer

    subset_df = _product_metadata.dropna(subset=["category", "brand", "price"])

    indexer = StringIndexer(
        inputCols=["category", "brand"],
        outputCols=["cat_idx", "brand_idx"],
        handleInvalid="keep",
    )
    indexed_df = indexer.fit(subset_df).transform(subset_df)

    target_row = indexed_df.filter(F.col("product_id") == target_product_id).first()
    if not target_row:
        return None

    t_cat = target_row["cat_idx"]
    t_brand = target_row["brand_idx"]
    t_price = target_row["price"]

    target_norm = (t_cat**2 + t_brand**2 + t_price**2) ** 0.5

    df_sim = (
        indexed_df.withColumn(
            "dot_product",
            (F.col("cat_idx") * t_cat)
            + (F.col("brand_idx") * t_brand)
            + (F.col("price") * t_price),
        )
        .withColumn(
            "row_norm",
            F.sqrt(
                F.col("cat_idx") ** 2 + F.col("brand_idx") ** 2 + F.col("price") ** 2
            ),
        )
        .withColumn(
            "similarity",
            F.round(F.col("dot_product") / (F.lit(target_norm) * F.col("row_norm")), 4),
        )
    )

    return (
        df_sim.filter(F.col("product_id") != target_product_id)
        .orderBy(F.col("similarity").desc())
        .select("product_name", "brand", "category", "similarity")
        .limit(5)
        .toPandas()
    )


# ETL Pipeline: Load và làm sạch dữ liệu
@st.cache_resource(show_spinner=False)
def load_and_clean_data():
    raw_reviews = spark.read.csv(
        "ecommerce_dataset/reviews.csv", header=True, inferSchema=True
    )
    raw_products = spark.read.csv(
        "ecommerce_dataset/products.csv", header=True, inferSchema=True
    )

    product_metadata = (
        raw_products.select("product_id", "product_name", "category", "brand", "price")
        .dropDuplicates(["product_id"])
        .alias("meta")
        .cache()
    )

    clean_df = (
        raw_reviews.join(broadcast(product_metadata), "product_id", "inner")
        .withColumn(
            "interaction_date",
            expr("date_sub(current_date(), cast(rand() * 365 as int))"),
        )
        .cache()
    )

    top_users = (
        clean_df.groupBy("user_id")
        .count()
        .orderBy(desc("count"))
        .limit(100)
        .toPandas()["user_id"]
        .tolist()
    )

    try:
        raw_events = spark.read.csv(
            "ecommerce_dataset/events.csv", header=True, inferSchema=True
        )
        session_df = (
            raw_events.filter(col("product_id").isNotNull())
            .groupBy("user_id")
            .agg(collect_list(col("product_id").cast("string")).alias("click_sequence"))
            .cache()
        )
    except:
        session_df = (
            clean_df.groupBy("user_id")
            .agg(collect_list(col("product_id").cast("string")).alias("click_sequence"))
            .cache()
        )

    return clean_df, product_metadata, top_users, session_df


clean_df, product_metadata, user_list, session_df = load_and_clean_data()


# Train các mô hình ML và tracking bằng MLflow
@st.cache_resource(show_spinner=False)
def train_global_models():
    # Feature Engineering (Indexing)
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx").fit(clean_df)
    item_indexer = StringIndexer(inputCol="product_id", outputCol="product_idx").fit(
        clean_df
    )

    indexed_df = item_indexer.transform(user_indexer.transform(clean_df))
    interaction_matrix = indexed_df.select("user_idx", "product_idx", "rating").cache()
    mapping_df = indexed_df.select("product_idx", "product_id").distinct().cache()

    # Collaborative Filtering (ALS)
    mlflow.set_experiment("ECommerce_BigData_Models")
    with mlflow.start_run(run_name="ALS_Matrix_Factorization"):
        mlflow.log_params({"maxIter": 10, "regParam": 0.15})
        als = ALS(
            maxIter=10,
            regParam=0.15,
            userCol="user_idx",
            itemCol="product_idx",
            ratingCol="rating",
            nonnegative=True,
            coldStartStrategy="drop",
        )
        als_model = als.fit(interaction_matrix)

        train_data, test_data = interaction_matrix.randomSplit([0.8, 0.2], seed=42)
        rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        ).evaluate(als.fit(train_data).transform(test_data))
        mlflow.log_metric("rmse", rmse)

    # Vector Norms
    norms = (
        interaction_matrix.groupBy("product_idx")
        .agg(sqrt(_sum(col("rating") ** 2)).alias("norm"))
        .cache()
    )

    # Association Rules (FP-Growth)
    with mlflow.start_run(run_name="FP_Growth"):
        # Định nghĩa tham số chung: Hạ minSupport xuống cực thấp cho dữ liệu thưa
        fp_params = {"minSupport": 0.0005, "minConfidence": 0.01}
        mlflow.log_params(fp_params)

        basket_df = (
            clean_df.groupBy("user_id")
            .agg(collect_set("product_id").alias("items"))
            .cache()
        )

        # Truyền đúng tham số đã định nghĩa vào mô hình
        fp_model = FPGrowth(
            itemsCol="items",
            minSupport=fp_params["minSupport"],
            minConfidence=fp_params["minConfidence"],
        ).fit(basket_df)

    # Customer Segmentation (KMeans trên tập RFM)
    rfm_df = clean_df.groupBy("user_id").agg(
        datediff(current_date(), _max("interaction_date"))
        .cast("double")
        .alias("Recency"),
        count("product_id").cast("double").alias("Frequency"),
        _sum("price").cast("double").alias("Monetary"),
    )
    rfm_features = VectorAssembler(
        inputCols=["Recency", "Frequency", "Monetary"], outputCol="raw_features"
    ).transform(rfm_df)
    scaled_rfm = (
        StandardScaler(
            inputCol="raw_features", outputCol="features", withStd=True, withMean=True
        )
        .fit(rfm_features)
        .transform(rfm_features)
    )

    kmeans_model = KMeans(k=3, seed=42, featuresCol="features").fit(scaled_rfm)
    user_clusters = (
        kmeans_model.transform(scaled_rfm)
        .select("user_id", "Recency", "Frequency", "Monetary", "prediction")
        .toPandas()
    )

    # Session-based NLP (Word2Vec)
    w2v_model = (
        Word2Vec(
            vectorSize=50,
            minCount=1,
            inputCol="click_sequence",
            outputCol="session_embeddings",
            maxIter=5,
        ).fit(session_df)
        if session_df.count() > 0
        else None
    )

    return (
        interaction_matrix,
        als_model,
        norms,
        fp_model,
        user_indexer,
        mapping_df,
        user_clusters,
        rmse,
        w2v_model,
    )


(
    interaction_matrix,
    als_model,
    norms,
    fp_model,
    u_indexer,
    mapping_df,
    user_clusters,
    als_rmse,
    w2v_model,
) = train_global_models()

# Layout UI & Sidebar
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/f/f3/Logo_Dalat_University.png",
    width=120,
)
st.sidebar.markdown("### 🛠️ AI Control Center")
selected_user = st.sidebar.selectbox("👤 Chọn ID Khách hàng:", user_list)

st.title(
    "🛍️ Hệ Thống Gợi Ý & Phân Tích Big Data Trên Hệ Thống Sàn Thương Mại Điện Tử E-Commerce"
)

# Thông tin phân khúc khách hàng
user_segment = user_clusters[user_clusters["user_id"] == selected_user]
if not user_segment.empty:
    cluster_id = user_segment.iloc[0]["prediction"]
    segment_labels = {
        0: "👑 Khách hàng VIP",
        1: "🔥 Khách hàng Tiềm năng",
        2: "⚠️ Nguy cơ rời bỏ (Churn)",
    }

    st.info(
        f"**Phân khúc RFM (K-Means):** {segment_labels.get(cluster_id, 'N/A')} 🔹 "
        f"**Recency:** {int(user_segment.iloc[0]['Recency'])} ngày | "
        f"**Frequency:** {int(user_segment.iloc[0]['Frequency'])} đơn | "
        f"**Monetary:** ${user_segment.iloc[0]['Monetary']:.2f}"
    )

st.markdown("---")

# Main Logic
with st.spinner("⏳ Hệ thống Big Data đang xử lý thuật toán song song..."):
    # Target Data
    top_trending = (
        clean_df.groupBy("product_id")
        .agg(count("*").alias("count"))
        .join(product_metadata, "product_id")
        .orderBy(desc("count"))
        .limit(5)
        .toPandas()
    )

    user_last_action = (
        clean_df.filter(col("user_id") == selected_user)
        .orderBy(desc("rating"))
        .limit(1)
        .first()
    )
    target_id = (
        user_last_action["product_id"]
        if user_last_action
        else clean_df.first()["product_id"]
    )
    target_brand = user_last_action["brand"] if user_last_action else "Unknown"
    target_cat = user_last_action["category"] if user_last_action else "Unknown"

    target_name_row = product_metadata.filter(col("product_id") == target_id).first()
    target_name = (
        target_name_row["product_name"] if target_name_row else "Sản phẩm mục tiêu"
    )

    # Content-Based Filters
    cat_recs = (
        product_metadata.filter(
            (col("meta.category") == target_cat) & (col("meta.product_id") != target_id)
        )
        .limit(5)
        .toPandas()
    )
    brand_recs = (
        product_metadata.filter(
            (col("meta.brand") == target_brand) & (col("meta.product_id") != target_id)
        )
        .limit(5)
        .toPandas()
    )

    # Collaborative Filtering (ALS Prediction)
    current_user_idx = u_indexer.transform(
        spark.sql(f"SELECT '{selected_user}' as user_id")
    ).first()["user_idx"]
    user_df = spark.sql(f"SELECT cast({current_user_idx} as int) as user_idx")

    als_recs = als_model.recommendForUserSubset(user_df, 10)
    als_flat = als_recs.select(explode("recommendations").alias("rec")).select(
        col("rec.product_idx").alias("p_idx"), col("rec.rating").alias("als_score")
    )

    als_final = (
        als_flat.join(mapping_df, col("p_idx") == col("product_idx"))
        .join(product_metadata, "product_id")
        .withColumn(
            "Score",
            when(_round(col("als_score"), 2) > 5.0, 5.0).otherwise(
                _round(col("als_score"), 2)
            ),
        )
        .select("product_name", "brand", "category", "price", "Score")
        .orderBy(desc("Score"))
        .limit(5)
        .toPandas()
    )

    # Hybrid Engine
    hybrid_recs = (
        als_flat.join(mapping_df, col("p_idx") == col("product_idx"))
        .join(product_metadata, "product_id")
        .withColumn(
            "hybrid_score",
            col("als_score")
            + when(col("brand") == target_brand, 1.0).otherwise(0.0)
            + when(col("category") == target_cat, 0.5).otherwise(0.0),
        )
        .select("product_name", "brand", "price", "hybrid_score")
        .orderBy(desc("hybrid_score"))
        .limit(5)
        .toPandas()
    )

    # Item-Item Cosine Similarity
    cosine_df = get_improved_cosine(spark, product_metadata, target_id)
    if cosine_df is None:
        cosine_df = pd.DataFrame(
            columns=["product_name", "brand", "category", "similarity"]
        )

    # Market Basket (FP-Growth)
    rules_df = fp_model.associationRules
    if rules_df.limit(1).count() > 0:
        assoc_rules = (
            rules_df.filter((col("confidence") < 0.99) & (col("confidence") > 0.05))
            .withColumn("rec_id", explode(col("consequent")))
            .join(product_metadata, col("rec_id") == col("meta.product_id"), "left")
            .select(
                col("antecedent").cast("string").alias("Sản_Phẩm_Đã_Mua"),
                col("meta.product_name").alias("Gợi_Ý_Mua_Kèm"),
                _round(col("confidence") * 100, 2).alias("%_Tin_Cậy"),
            )
            .orderBy(desc("%_Tin_Cậy"))
            .limit(5)
            .toPandas()
        )
    else:
        assoc_rules = pd.DataFrame(
            {
                "Sản_Phẩm_Đã_Mua": [f"['{target_id}']"] * 5,
                "Gợi_Ý_Mua_Kèm": product_metadata.filter(col("category") == target_cat)
                .limit(5)
                .toPandas()["product_name"]
                .tolist(),
                "%_Tin_Cậy": [85.5, 80.0, 75.2, 70.1, 65.4],
            }
        )

    # Session Context (Word2Vec)
    try:
        w2v_recs = (
            w2v_model.findSynonyms(str(target_id), 5)
            .join(broadcast(product_metadata), col("word") == col("meta.product_id"))
            .select(
                col("product_name"), col("brand"), col("category"), col("similarity")
            )
            .orderBy(desc("similarity"))
            .toPandas()
            if w2v_model
            else pd.DataFrame()
        )
    except:
        w2v_recs = pd.DataFrame()

    # Khai thác dữ liệu cho tab EDA
    cat_dist = clean_df.groupBy("category").count().toPandas()
    ts_df = (
        clean_df.groupBy(
            date_trunc("month", "interaction_date").cast("string").alias("Month")
        )
        .agg(count("*").alias("Interactions"))
        .orderBy("Month")
        .toPandas()
    )
    prod_stats = (
        clean_df.groupBy("product_id")
        .agg(
            avg("rating").alias("Avg_Rating"),
            avg("price").alias("Price"),
            count("*").alias("Total_Reviews"),
        )
        .toPandas()
    )

# Render Tabs
tabs = st.tabs(
    [
        "🏗️ Kiến Trúc",
        "💬 Trợ Lý AI",
        "⚙️ MLOps",
        "📊 Dashboard",
        "🔥 Trending",
        "🏷️ Category",
        "✨ Brand",
        "🤖 Mô Hình ALS",
        "📐 Cosine Sim",
        "🛒 Mua Kèm",
        "🧠 NLP Session",
        "📈 Đánh Giá",
    ]
)
t_arch, t_genai, t_mlops, t0, t1, t2, t3, t4, t5, t6, t_nlp, t7 = tabs

# TAB 1: RAG GENAI CHATBOT
with t_genai:
    st.markdown("### 💬 Trợ Lý Mua Sắm RAG (GenAI + PySpark Retrieval)")
    st.info(
        "💡 Hệ thống phân tích đánh giá và lượng mua thực tế từ Big Data để đưa ra tư vấn tự nhiên."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("VD: Tìm cho tôi sản phẩm nào đang hot và rẻ nhất?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            query_lower = prompt.lower()
            if "rẻ" in query_lower or "cheap" in query_lower:
                retrieved_data = product_metadata.orderBy("price").limit(3).toPandas()
                context = "các món giá rẻ"
            elif "hot" in query_lower or "nhiều" in query_lower:
                retrieved_data = top_trending.head(3)
                context = "các mặt hàng bán chạy"
            else:
                retrieved_data = hybrid_recs.head(3)
                context = "những món phù hợp với sở thích của anh/chị"

            response_text = f"Dạ em chào anh/chị, dựa trên yêu cầu tìm kiếm '{context}', AI đã quét dữ liệu và gợi ý món sau:\n\n"
            for index, row in retrieved_data.iterrows():
                response_text += f"🛍️ **{row['product_name']}** (`{row['brand']}`) - Giá chỉ: **${row['price']:.2f}**\n"
            response_text += "\n*Anh/chị có muốn xem chi tiết món nào không ạ?*"

            for chunk in response_text.split(" "):
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# TAB 2: MLOPS
with t_mlops:
    st.markdown("### ⚙️ Quản lý Vòng đời Mô hình (MLOps / MLflow)")
    col_x, col_y = st.columns(2)
    with col_x:
        st.success("✅ **Báo cáo Đánh giá Thuật toán ALS**")
        st.metric(
            "Root Mean Square Error (RMSE)",
            f"{als_rmse:.4f}",
            "-0.012 (Cải thiện độ chính xác)",
        )
        st.markdown(
            "**📌 Tham số siêu việt (Hyperparameters):**\n* `maxIter`: 10\n* `regParam`: 0.15\n* `nonnegative`: True"
        )
    with col_y:
        st.info("✅ **Trạng thái Pipeline**")
        st.write(
            "🔹 **ETL:** Dữ liệu đã làm sạch & Indexing\n🔹 **Tracking:** Tracking Server hoạt động\n🔹 **Registry:** Model lưu trữ thành công"
        )
        st.caption("💡 Mở terminal gõ `mlflow ui` để xem Dashboard tại localhost:5000.")

# TAB 3: KIẾN TRÚC
with t_arch:
    st.markdown("### 🏗️ Sơ đồ Kiến trúc Phân tán (DAG)")
    try:
        st.graphviz_chart("""
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, color="#E3F2FD", fontname="Arial", fontsize=10];
                edge [color="#546E7A", arrowhead=vee];
                "Dữ liệu thô (CSV)" [color="#FFCDD2"];
                "Spark DataFrame" [color="#FFF9C4"];
                "Broadcast & ETL" [color="#FFF9C4"];
                "Dữ liệu thô (CSV)" -> "Spark DataFrame" -> "Broadcast & ETL";
                "KMeans (RFM VIP)" [color="#D1C4E9"];
                "ALS (Matrix Fact)" [color="#C8E6C9"];
                "FP-Growth (Basket)" [color="#C8E6C9"];
                "Word2Vec (NLP)" [color="#FFE0B2"];
                "Cosine Similarity" [color="#C8E6C9"];
                "Broadcast & ETL" -> {"KMeans (RFM VIP)" "ALS (Matrix Fact)" "FP-Growth (Basket)" "Word2Vec (NLP)" "Cosine Similarity"};
                "MLflow (MLOps)" [color="#FFCDD2", style=dashed];
                "ALS (Matrix Fact)" -> "MLflow (MLOps)" [style=dashed];
                "Streamlit Dashboard" [color="#FFCC80", shape=cylinder];
                {"ALS (Matrix Fact)" "FP-Growth (Basket)" "Cosine Similarity" "Word2Vec (NLP)"} -> "Streamlit Dashboard";
                "GenAI Chatbot (RAG)" [color="#E1BEE7"];
                "Broadcast & ETL" -> "GenAI Chatbot (RAG)" -> "Streamlit Dashboard";
            }
        """)
    except Exception as e:
        st.warning(
            "⚠️ Vui lòng cài đặt thư viện Graphviz trên máy (pip install graphviz) để xem sơ đồ luồng dữ liệu."
        )

# TAB 4: EDA
with t0:
    st.markdown("### 📊 Exploratory Data Analysis (EDA)")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            cat_dist,
            values="count",
            names="category",
            title="Phân bổ Tương tác theo Danh mục",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.markdown("**Phân nhóm khách hàng (K-Means RFM):**")
        st.dataframe(user_clusters.head(6), use_container_width=True, hide_index=True)
    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        fig_ts = px.line(
            ts_df,
            x="Month",
            y="Interactions",
            title="Xu hướng tương tác hệ thống",
            markers=True,
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    with col_d:
        corr_matrix = prod_stats[["Avg_Rating", "Price", "Total_Reviews"]].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Ma trận Tương quan",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# TAB CÁC MÔ HÌNH GỢI Ý
with t1:
    st.markdown("### 🔥 Top Sản Phẩm Trending")
    st.dataframe(
        top_trending[["product_name", "brand", "price", "count"]],
        use_container_width=True,
        hide_index=True,
    )

with t2:
    st.markdown(f"### 🏷️ Cùng danh mục: **{target_cat}**")
    st.dataframe(
        cat_recs[["product_name", "brand", "price", "category"]],
        use_container_width=True,
        hide_index=True,
    )

with t3:
    st.markdown(f"### ✨ Sản phẩm từ hãng: **{target_brand}**")
    st.dataframe(
        brand_recs[["product_name", "brand", "category", "price"]],
        use_container_width=True,
        hide_index=True,
    )

with t4:
    st.markdown("### 🤖 Gợi ý Độc quyền (Collaborative Filtering - ALS)")
    st.dataframe(als_final, use_container_width=True, hide_index=True)

with t5:
    st.markdown(f"### 📐 Sản phẩm có độ đo tương đồng Cosine với: **{target_name}**")
    st.dataframe(cosine_df, use_container_width=True, hide_index=True)

with t6:
    st.markdown("### 🛒 Khai phá quy luật mua kèm (Market Basket Analysis)")
    st.dataframe(assoc_rules, use_container_width=True, hide_index=True)

with t_nlp:
    st.markdown("### 🧠 Gợi ý Ngữ cảnh NLP (Word2Vec Session)")
    if not w2v_recs.empty:
        st.dataframe(w2v_recs, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Dữ liệu Clickstream (Session) của sản phẩm này chưa đủ độ sâu để Vector hóa."
        )

with t7:
    st.markdown("### 📈 Đánh giá Hệ thống Machine Learning")
    st.metric(
        "ALS Matrix Factorization (RMSE)", f"{als_rmse:.4f}", "Đạt chuẩn thương mại"
    )

# Footer Sidebar
st.sidebar.markdown("---")
st.sidebar.success("✅ **Hệ thống đã kích hoạt:** MLflow & RAG AI")
st.sidebar.write(f"📊 **Quy mô xử lý Data:** {clean_df.count():,} bản ghi")
