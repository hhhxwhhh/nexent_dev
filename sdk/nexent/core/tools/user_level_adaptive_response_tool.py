import json
import logging
import re
from typing import Dict, Any
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
                "图片", "报告", "结果", "诊断", "恶性", "良性", "转移"
            ],
            "intermediate": [
                "病理类型", "分化程度", "浸润", "转移", "淋巴结", "分期", "分级", "预后",
                "免疫组化", "Ki67", "P53", "ER", "PR", "HER2", "组织学", "形态学", 
                "核分裂", "异型性", "腺体", "间质", "血管", "神经", "边界", "包膜",
                "机制", "原因", "关系", "区别", "特征", "表现", "临床意义"
            ],
            "advanced": [
                "分子病理", "基因突变", "信号通路", "蛋白表达", "靶向治疗", "耐药",
                "微环境", "免疫检查点", "PD-L1", "BRCA", "EGFR", "ALK", "ROS1",
                "KRAS", "NRAS", "BRAF", "MSI", "TMB", "循环肿瘤", "ctDNA", "外显子",
                "转录组", "蛋白质组", "表观遗传", "甲基化", "microRNA", "lncRNA",
                "最新研究", "前沿进展", "文献", "指南", "共识", "标准", "规范"
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
                "caution": "强调需要专业医生诊断"
            },
            "intermediate": {
                "language_style": "专业但易懂",
                "term_usage": "正常使用病理学术语，关键术语稍作解释",
                "explanation_depth": "中等深度，讲解基本机制和临床意义",
                "analogy_use": "适度使用类比，重点放在专业内容上",
                "structure": "先给出准确定义，解释基本原理，说明临床应用",
                "caution": "提醒AI辅助诊断的局限性"
            },
            "advanced": {
                "language_style": "高度专业化",
                "term_usage": "自由使用专业术语，无需过多解释",
                "explanation_depth": "深入机制，引用最新研究",
                "analogy_use": "较少使用类比，专注于技术细节",
                "structure": "先给出精确定义，深入分析机制，介绍研究前沿",
                "caution": "指出争议点和研究空白"
            }
        }
        
        logger.info("PathologyUserLevelAdaptiveResponseTool initialized")

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "microscope", "text": "分析用户病理学认知水平"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _assess_user_level(self, question: str, context: str = None) -> Dict[str, Any]:
        """评估用户病理学认知水平
        
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
        
        strategy = f"根据您的问题，我判断您的病理学知识水平为{user_level}级别。\n\n"
        strategy += f"我将以{template['language_style']}的方式为您解答：\n\n"
        
        # 添加回答策略说明
        strategy += "回答策略：\n"
        strategy += f"- 语言风格：{template['language_style']}\n"
        strategy += f"- 术语使用：{template['term_usage']}\n"
        strategy += f"- 解释深度：{template['explanation_depth']}\n"
        strategy += f"- 类比运用：{template['analogy_use']}\n"
        strategy += f"- 回答结构：{template['structure']}\n"
        strategy += f"- 注意事项：{template['caution']}\n\n"
        
        # 添加置信度信息
        strategy += f"水平判断置信度：{confidence:.2%}\n\n"
        
        return strategy

    def forward(
        self, 
        question: str, 
        context: str = None
    ) -> str:
        """根据用户病理学问题判断其认知水平并提供相应的回答策略。
        
        Args:
            question (str): 用户提出的病理学相关问题
            context (str): 问题的相关背景信息或上下文
            
        Returns:
            str: JSON格式的用户认知水平评估和自适应回答策略
        """
        try:
            # 发送工具运行消息
            self._send_tool_message()

            # 评估用户认知水平
            user_level_assessment = self._assess_user_level(question, context)
            
            # 生成自适应回答策略
            adaptive_response_strategy = self._generate_adaptive_response_strategy(user_level_assessment)
            
            # 准备结果
            result = {
                "status": "success",
                "user_pathology_level": user_level_assessment["user_level"],
                "confidence": user_level_assessment["confidence"],
                "assessment_metrics": user_level_assessment["metrics"],
                "adaptive_response_strategy": adaptive_response_strategy,
                "message": f"用户病理学认知水平评估完成，判定为{user_level_assessment['user_level']}级别"
            }
            
            logger.info(f"User pathology level assessment completed: {user_level_assessment['user_level']} "
                       f"(confidence: {user_level_assessment['confidence']:.2%})")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in pathology user level adaptive response: {str(e)}", exc_info=True)
            error_msg = f"病理学用户水平自适应响应处理失败: {str(e)}"
            raise Exception(error_msg)