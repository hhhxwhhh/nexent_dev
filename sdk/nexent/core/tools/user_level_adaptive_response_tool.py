import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("user_level_adaptive_response_tool")


class PathologyUserLevelAdaptiveResponseTool(Tool):
    """病理学用户水平自适应响应工具，根据用户问题判断其病理学认知水平并提供相应复杂度的回答"""
    
    name = "pathology_user_level_adaptive_response"
    description = (
        "智能评估用户病理学认知水平的工具，通过分析用户提出的病理学相关问题来判断其专业知识水平，"
        "并根据用户的理解能力提供适当复杂度和专业程度的病理学解答。"
        "适用于病理学教育、诊断辅助、病例讨论等场景。"
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "用户提出的病理学相关问题"
        },
        "context": {
            "type": "string",
            "description": "问题的相关背景信息或上下文（如病例信息、图像描述等）",
            "nullable": True
        },
        "conversation_history": {
            "type": "array",
            "description": "用户与系统的对话历史，用于更准确地评估用户水平",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "发言者角色（user或assistant）"},
                    "content": {"type": "string", "description": "发言内容"}
                }
            },
            "nullable": True
        },
        "preferred_complexity": {
            "type": "string",
            "description": "用户偏好的解释复杂度（beginner/intermediate/advanced），如果用户明确指定",
            "enum": ["beginner", "intermediate", "advanced"],
            "nullable": True
        }
    }
    output_type = "string"
    category = ToolCategory.TEXT_PROCESSING.value
    tool_sign = ToolSign.ANALYSIS_OPERATION.value

    def __init__(
        self,
        observer: MessageObserver = Field(description="消息观察者", default=None, exclude=True)
    ):
        """初始化病理学用户水平自适应响应工具。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
        """
        super().__init__()
        self.observer = observer
        self.running_prompt_zh = "正在分析用户病理学认知水平..."
        self.running_prompt_en = "Analyzing user's pathology cognitive level..."
        
        # 病理学领域关键词分类
        self.pathology_keywords = {
            "beginner": [
                "癌症", "肿瘤", "细胞", "组织", "切片", "染色", "HE", "什么叫", "什么是", 
                "意思", "解释", "不懂", "不明白", "简单说", "通俗", "怎么看", "如何看",
                "图片", "报告", "结果", "诊断", "恶性", "良性", "转移", "检查", "化验"
            ],
            "intermediate": [
                "病理类型", "分化程度", "浸润", "转移", "淋巴结", "分期", "分级", "预后",
                "免疫组化", "Ki67", "P53", "ER", "PR", "HER2", "组织学", "形态学", 
                "核分裂", "异型性", "腺体", "间质", "血管", "神经", "边界", "包膜",
                "机制", "原因", "关系", "区别", "特征", "表现", "临床意义", "治疗",
                "手术", "化疗", "放疗", "靶向"
            ],
            "advanced": [
                "分子病理", "基因突变", "信号通路", "蛋白表达", "靶向治疗", "耐药",
                "微环境", "免疫检查点", "PD-L1", "BRCA", "EGFR", "ALK", "ROS1",
                "KRAS", "NRAS", "BRAF", "MSI", "TMB", "循环肿瘤", "ctDNA", "外显子",
                "转录组", "蛋白质组", "表观遗传", "甲基化", "microRNA", "lncRNA",
                "最新研究", "前沿进展", "文献", "指南", "共识", "标准", "规范",
                "临床试验", "生物标志物", "个体化医疗", "精准医学"
            ]
        }
        
        # 不同水平用户的回答模板
        self.response_templates = {
            "beginner": {
                "language_style": "通俗易懂",
                "term_usage": "尽量避免专业术语，必要时给予解释",
                "explanation_depth": "浅层解释，重点讲是什么和有什么用",
                "analogy_use": "多使用生活中的类比帮助理解",
                "structure": "先给出简单定义，再用类比解释，最后说明意义",
                "caution": "强调需要专业医生诊断",
                "examples": "提供常见例子帮助理解"
            },
            "intermediate": {
                "language_style": "专业但易懂",
                "term_usage": "正常使用病理学术语，关键术语稍作解释",
                "explanation_depth": "中等深度，讲解基本机制和临床意义",
                "analogy_use": "适度使用类比，重点放在专业内容上",
                "structure": "先给出准确定义，解释基本原理，说明临床应用",
                "caution": "提醒AI辅助诊断的局限性",
                "examples": "结合典型病例说明"
            },
            "advanced": {
                "language_style": "高度专业化",
                "term_usage": "自由使用专业术语，无需过多解释",
                "explanation_depth": "深入机制，引用最新研究",
                "analogy_use": "较少使用类比，专注于技术细节",
                "structure": "先给出精确定义，深入分析机制，介绍研究前沿",
                "caution": "指出争议点和研究空白",
                "examples": "引用研究数据和文献"
            }
        }
        
        # 用户水平的历史权重因子
        self.history_weight_factor = 0.3
        
        logger.info("PathologyUserLevelAdaptiveResponseTool initialized")

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "microscope", "text": "分析用户病理学认知水平"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _assess_user_level_from_question(self, question: str, context: str = None) -> Dict[str, Any]:
        """基于问题本身评估用户病理学认知水平
        
        Args:
            question (str): 用户的问题
            context (str): 问题的上下文
            
        Returns:
            Dict[str, Any]: 用户认知水平评估结果
        """
        text_to_analyze = (question or "") + " " + (context or "")
        text_to_analyze = text_to_analyze.lower()
        
        # 计算各水平关键词出现次数
        scores = {}
        for level, keywords in self.pathology_keywords.items():
            scores[level] = sum(text_to_analyze.count(kw.lower()) for kw in keywords)
        
        # 特殊规则调整
        # 如果问题中包含问号且长度较短，倾向于初学者水平
        if '?' in question and len(question) < 20:
            scores["beginner"] += 2
            
        # 如果包含较长的专业词汇组合，倾向于高级水平
        if re.search(r'[a-zA-Z]{4,}-?[a-zA-Z]{4,}', text_to_analyze):
            scores["advanced"] += 2
            
        # 如果提及具体基因名或蛋白名，倾向于高级水平
        advanced_genes_proteins = ["p53", "ki67", "her2", "er", "pr", "egfr", "alk", "ros1", "braf", "kras", "nras", "brca", "msi", "tmb"]
        for gene in advanced_genes_proteins:
            if gene in text_to_analyze.lower():
                scores["advanced"] += 1
        
        # 判断用户水平
        total_score = sum(scores.values())
        if total_score == 0:
            # 如果没有找到关键词，默认为中级水平
            user_level = "intermediate"
            confidence = 0.5
        else:
            # 根据最高得分判断用户水平
            user_level = max(scores, key=scores.get)
            confidence = scores[user_level] / total_score
                
        # 考虑问题长度（较长的问题通常来自更有经验的用户）
        length_factor = min(len(question) / 100.0, 1.0)  # 归一化到0-1之间
        if length_factor > 0.7 and user_level == "beginner":
            user_level = "intermediate"
        
        return {
            "user_level": user_level,
            "confidence": confidence,
            "metrics": {
                "beginner_score": scores["beginner"],
                "intermediate_score": scores["intermediate"],
                "advanced_score": scores["advanced"],
                "length_factor": length_factor
            }
        }

    def _assess_user_level_from_history(self, conversation_history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """基于对话历史评估用户病理学认知水平
        
        Args:
            conversation_history (List[Dict[str, str]]): 对话历史
            
        Returns:
            Optional[Dict[str, Any]]: 基于历史的用户病理学认知水平评估结果
        """
        if not conversation_history:
            return None
            
        # 分析最近几次对话
        recent_interactions = conversation_history[-5:]  # 分析最近5次交互
        
        level_scores = {"beginner": 0, "intermediate": 0, "advanced": 0}
        total_interactions = 0
        
        for interaction in recent_interactions:
            if interaction["role"] == "user":
                # 对用户问题进行分析
                question_analysis = self._assess_user_level_from_question(interaction["content"])
                level_scores[question_analysis["user_level"]] += 1
                total_interactions += 1
            elif interaction["role"] == "assistant":
                assistant_response = interaction["content"]
                response_analysis = self._assess_user_level_from_question(assistant_response)
                

                if response_analysis["user_level"] == "advanced":
                    level_scores["intermediate"] += 0.5
                    level_scores["advanced"] += 1
                elif response_analysis["user_level"] == "intermediate":
                    level_scores["intermediate"] += 1
                else:  # beginner level response
                    level_scores["beginner"] += 1
                    
                total_interactions += 1
                
        # 确定主要水平
        if total_interactions == 0:
            return None
            
        # 计算每个级别的加权分数
        weighted_scores = {
            "beginner": level_scores["beginner"] / total_interactions,
            "intermediate": level_scores["intermediate"] / total_interactions,
            "advanced": level_scores["advanced"] / total_interactions
        }
        
        dominant_level = max(weighted_scores, key=weighted_scores.get)
        history_confidence = weighted_scores[dominant_level]
        
        return {
            "user_level": dominant_level,
            "confidence": history_confidence,
            "metrics": {
                "beginner": weighted_scores["beginner"],
                "intermediate": weighted_scores["intermediate"],
                "advanced": weighted_scores["advanced"],
                "total_interactions": total_interactions
            }
        }

    def _combine_assessments(self, question_assessment: Dict[str, Any], 
                           history_assessment: Optional[Dict[str, Any]],
                           preferred_complexity: Optional[str]) -> Dict[str, Any]:
        """结合多个评估结果得出最终用户水平判断
        
        Args:
            question_assessment (Dict[str, Any]): 基于问题的评估
            history_assessment (Optional[Dict[str, Any]]): 基于历史的评估
            preferred_complexity (Optional[str]): 用户偏好的复杂度
            
        Returns:
            Dict[str, Any]: 综合评估结果
        """
        # 如果用户明确指定了偏好复杂度，优先使用
        if preferred_complexity:
            return {
                "user_level": preferred_complexity,
                "confidence": 0.9,  # 高置信度
                "source": "user_preference"
            }
        
        # 结合问题评估和历史评估
        if history_assessment:
            # 加权平均两种评估结果
            question_weight = 1.0 - self.history_weight_factor
            history_weight = self.history_weight_factor
            
            # 计算加权得分
            combined_scores = {}
            for level in ["beginner", "intermediate", "advanced"]:
                question_score = question_assessment["metrics"].get(f"{level}_score", 0) if "metrics" in question_assessment else 0
                history_score = history_assessment["metrics"].get(level, 0) if "metrics" in history_assessment else 0
                combined_scores[level] = question_score * question_weight + history_score * history_weight
            
            final_level = max(combined_scores, key=combined_scores.get)
            final_confidence = (question_assessment["confidence"] * question_weight + 
                              history_assessment["confidence"] * history_weight)
            
            return {
                "user_level": final_level,
                "confidence": final_confidence,
                "source": "combined_assessment"
            }
        else:
            # 仅使用问题评估
            return {
                "user_level": question_assessment["user_level"],
                "confidence": question_assessment["confidence"],
                "source": "question_only"
            }

    def _generate_adaptive_response_strategy(self, user_level_assessment: Dict[str, Any]) -> str:
        """生成适合用户认知水平的回答策略
        
        Args:
            user_level_assessment (Dict[str, Any]): 用户认知水平评估结果
            
        Returns:
            str: 回答策略描述
        """
        user_level = user_level_assessment["user_level"]
        confidence = user_level_assessment["confidence"]
        template = self.response_templates[user_level]
        
        strategy = f"根据您的问题和交互历史，我判断您的病理学知识水平为{user_level}级别。\n\n"
        strategy += f"我将以{template['language_style']}的方式为您解答：\n\n"
        
        # 添加回答策略说明
        strategy += "回答策略：\n"
        strategy += f"- 语言风格：{template['language_style']}\n"
        strategy += f"- 术语使用：{template['term_usage']}\n"
        strategy += f"- 解释深度：{template['explanation_depth']}\n"
        strategy += f"- 类比运用：{template['analogy_use']}\n"
        strategy += f"- 回答结构：{template['structure']}\n"
        strategy += f"- 注意事项：{template['caution']}\n"
        strategy += f"- 示例说明：{template['examples']}\n\n"
        
        # 添加置信度信息
        strategy += f"水平判断置信度：{confidence:.2%}\n\n"
        
        return strategy

    def forward(
        self, 
        question: str, 
        context: str = None,
        conversation_history: List[Dict[str, str]] = None,
        preferred_complexity: str = None
    ) -> str:
        """根据用户病理学问题判断其认知水平并提供相应的回答策略。
        
        Args:
            question (str): 用户提出的病理学相关问题
            context (str): 问题的相关背景信息或上下文
            conversation_history (List[Dict[str, str]]): 用户与系统的对话历史
            preferred_complexity (str): 用户偏好的解释复杂度
            
        Returns:
            str: JSON格式的用户认知水平评估和自适应回答策略
        """
        try:
            # 发送工具运行消息
            self._send_tool_message()

            # 基于问题评估用户认知水平
            question_assessment = self._assess_user_level_from_question(question, context)
            
            # 基于对话历史评估用户认知水平
            history_assessment = self._assess_user_level_from_history(conversation_history)
            
            # 综合评估结果
            user_level_assessment = self._combine_assessments(
                question_assessment, 
                history_assessment,
                preferred_complexity
            )
            
            # 生成自适应回答策略
            adaptive_response_strategy = self._generate_adaptive_response_strategy(user_level_assessment)
            
            # 准备结果
            result = {
                "status": "success",
                "user_pathology_level": user_level_assessment["user_level"],
                "confidence": user_level_assessment["confidence"],
                "assessment_source": user_level_assessment.get("source", "unknown"),
                "question_assessment": question_assessment,
                "history_assessment": history_assessment,
                "adaptive_response_strategy": adaptive_response_strategy,
                "message": f"用户病理学认知水平评估完成，判定为{user_level_assessment['user_level']}级别"
            }
            
            logger.info(f"User pathology level assessment completed: {user_level_assessment['user_level']} "
                       f"(confidence: {user_level_assessment['confidence']:.2%}, source: {user_level_assessment.get('source', 'unknown')})")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in pathology user level adaptive response: {str(e)}", exc_info=True)
            error_msg = f"病理学用户水平自适应响应处理失败: {str(e)}"
            raise Exception(error_msg)