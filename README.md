## How to Diversify \texttt{any} Personalized Recommender? 

In this paper, we introduce a novel approach to improve the diversity of Top-N recommendations while maintaining recommendation performance. Our approach employs a user-centric pre-processing strategy aimed at exposing users to a wide array of content categories and topics. We personalize this strategy by selectively adding and removing a percentage of interactions from user profiles. This personalization ensures we remain closely aligned with user preferences while gradually introducing distribution shifts. 
Our pre-processing technique offers flexibility and can seamlessly integrate into any recommender architecture. We run extensive experiments on two publicly available data sets for news and book recommendations to evaluate our approach. We test various standard and neural network-based recommender system algorithms.
Our results show that our approach generates diverse recommendations, ensuring users are exposed to a wider range of items. Furthermore, leveraging pre-processed data for training leads to recommender systems achieving performance levels comparable to, and in some cases, better than those trained on original, unmodified data.
Additionally, our approach promotes provider fairness by facilitating exposure to minority or niche categories. 

![Diagram](Diagram.png)
