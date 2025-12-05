import json
import logging
from typing import Dict, Any, List
from pydantic import Field
from smolagents.tools import Tool

from ..utils.observer import MessageObserver, ProcessType
from ..utils.tools_common_message import ToolCategory, ToolSign

logger = logging.getLogger("user_level_adaptive_response_tool")


class UserLevelAdaptiveResponseTool(Tool):
    """用户水平自适应响应工具，根据用户问题判断其认知水平并提供相应复杂度的回答"""
    
    name = "user_level_adaptive_response"
    description = (
        "智能评估用户认知水平的工具，通过分析用户提出的问题来判断其专业知识水平，"
        "并根据用户的理解能力提供适当复杂度的回答。适用于所有需要个性化教育和解释的场景。"
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "用户提出的问题"
        },
        "context": {
            "type": "string",
            "description": "问题的相关背景信息或上下文",
            "nullable": True
        },
        "domain": {
            "type": "string",
            "description": "问题所属的专业领域（如病理学、医学、计算机科学等）",
            "default": "general",
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
        """初始化用户水平自适应响应工具。
        
        Args:
            observer (MessageObserver, optional): 消息观察者实例。默认为 None。
        """
        super().__init__()
        self.observer = observer
        self.running_prompt_zh = "正在分析用户认知水平..."
        self.running_prompt_en = "Analyzing user cognitive level..."
        
        logger.info("UserLevelAdaptiveResponseTool initialized")

    def _send_tool_message(self):
        """发送工具运行消息"""
        if self.observer:
            running_prompt = self.running_prompt_zh if self.observer.lang == "zh" else self.running_prompt_en
            self.observer.add_message("", ProcessType.TOOL, running_prompt)
            card_content = [{"icon": "user-check", "text": "分析用户认知水平"}]
            self.observer.add_message("", ProcessType.CARD, json.dumps(card_content, ensure_ascii=False))

    def _assess_user_level(self, question: str, context: str = None, domain: str = "general") -> Dict[str, Any]:
        """评估用户认知水平
        
        Args:
            question (str): 用户的问题
            context (str): 问题的上下文
            domain (str): 问题所属领域
            
        Returns:
            Dict[str, Any]: 用户认知水平评估结果
        """
        # 关键词列表用于判断用户水平
        beginner_keywords = [
            "是什么", "什么意思", "怎么理解", "什么是", "简单说", "通俗", "不明白", 
            "不懂", "解释一下", "怎么", "能否", "可以", "如何", "请问", "求教"
        ]
        
        intermediate_keywords = [
            "机制", "原理", "关系", "区别", "联系", "影响", "作用", "功能", 
            "过程", "方法", "技术", "应用", "表现", "特点", "特征", "条件"
        ]
        
        advanced_keywords = [
            "深入", "详细", "具体", "深度", "分子", "细胞", "病理", "生化", 
            "机制研究", "临床意义", "最新进展", "前沿", "研究", "论文", 
            "实验", "数据", "统计", "分析", "诊断", "治疗", "预后"
        ]
        
        # 领域特定关键词
        domain_keywords = {
            "pathology": ["病理", "组织", "细胞", "肿瘤", "癌", "病变", "切片", "染色", "HE", "免疫组化"],
            "medicine": ["疾病", "症状", "诊断", "治疗", "药物", "病人", "临床", "检查"],
            "computer_science": ["算法", "程序", "代码", "系统", "网络", "数据库", "计算", "编程"]
        }
        
        # 计算各水平关键词出现次数
        beginner_score = sum(1 for kw in beginner_keywords if kw in question or (context and kw in context))
        intermediate_score = sum(1 for kw in intermediate_keywords if kw in question or (context and kw in context))
        advanced_score = sum(1 for kw in advanced_keywords if kw in question or (context and kw in context))
        
        # 计算领域相关性得分
        domain_score = 0
        if domain in domain_keywords:
            domain_score = sum(1 for kw in domain_keywords[domain] if kw in question or (context and kw in context))
        
        # 判断用户水平
        total_score = beginner_score + intermediate_score + advanced_score
        if total_score == 0:
            # 如果没有找到关键词，默认为中级水平
            user_level = "intermediate"
            confidence = 0.5
        else:
            # 根据得分比例判断用户水平
            beginner_ratio = beginner_score / total_score
            intermediate_ratio = intermediate_score / total_score
            advanced_ratio = advanced_score / total_score
            
            if beginner_ratio >= 0.5:
                user_level = "beginner"
                confidence = beginner_ratio
            elif advanced_ratio >= 0.4:
                user_level = "advanced"
                confidence = advanced_ratio
            else:
                user_level = "intermediate"
                confidence = intermediate_ratio
                
        # 考虑问题长度（较长的问题通常来自更有经验的用户）
        length_factor = min(len(question) / 100.0, 1.0)  # 归一化到0-1之间
        if length_factor > 0.7 and user_level == "beginner":
            user_level = "intermediate"
        
        return {
            "user_level": user_level,
            "confidence": confidence,
            "domain_relevance": domain_score > 0,
            "metrics": {
                "beginner_score": beginner_score,
                "intermediate_score": intermediate_score,
                "advanced_score": advanced_score,
                "domain_score": domain_score,
                "length_factor": length_factor
            }
        }

    def _generate_adaptive_response(self, question: str, user_level_assessment: Dict[str, Any], domain: str = "general") -> str:
        """生成适合用户认知水平的回答
        
        Args:
            question (str): 用户的问题
            user_level_assessment (Dict[str, Any]): 用户认知水平评估结果
            domain (str): 问题所属领域
            
        Returns:
            str: 适合用户水平的回答
        """
        user_level = user_level_assessment["user_level"]
        confidence = user_level_assessment["confidence"]
        
        response_templates = {
            "beginner": {
                "style": "通俗易懂",
                "depth": "浅层解释",
                "detail": "使用日常语言，避免专业术语，通过比喻和例子帮助理解",
                "structure": "先给出简单定义，再用生活中的类比解释，最后提供简单的应用场景"
            },
            "intermediate": {
                "style": "平衡专业性与易懂性",
                "depth": "中等深度解释",
                "detail": "适度使用专业术语，提供基本原理和工作机制，结合实际应用",
                "structure": "先给出准确定义，解释基本原理，提供相关应用和意义"
            },
            "advanced": {
                "style": "专业深入",
                "depth": "深层技术细节",
                "detail": "使用专业术语，提供详细的技术机制和最新研究进展，结合数据分析",
                "structure": "先给出精确定义，深入分析机制，提供研究现状和未来趋势"
            }
        }
        
        template = response_templates[user_level]
        
        response = f"根据您的问题，我判断您的{'病理学' if domain=='pathology' else '相关'}知识水平为{template['depth']}。\n\n"
        response += f"我将以{template['style']}的方式为您解答：\n\n"
        
        # 添加回答策略说明
        response += f"回答策略：{template['structure']}\n\n"
        
        # 添加置信度信息
        response += f"水平判断置信度：{confidence:.2%}\n\n"
        
        # 占位符，实际应用中这里会接入真实的回答生成逻辑
        response += "[此处将插入针对您问题的具体回答内容]"
        
        return response

    def forward(
        self, 
        question: str, 
        context: str = None,
        domain: str = "general"
    ) -> str:
        """根据用户问题判断其认知水平并提供相应的回答。
        
        Args:
            question (str): 用户提出的问题
            context (str): 问题的相关背景信息或上下文
            domain (str): 问题所属的专业领域
            
        Returns:
            str: JSON格式的用户认知水平评估和自适应回答策略
        """
        try:
            # 发送工具运行消息
            self._send_tool_message()

            # 评估用户认知水平
            user_level_assessment = self._assess_user_level(question, context, domain)
            
            # 生成自适应回答策略
            adaptive_response = self._generate_adaptive_response(question, user_level_assessment, domain)
            
            # 准备结果
            result = {
                "status": "success",
                "user_level": user_level_assessment["user_level"],
                "confidence": user_level_assessment["confidence"],
                "domain_relevance": user_level_assessment["domain_relevance"],
                "assessment_metrics": user_level_assessment["metrics"],
                "adaptive_response_strategy": adaptive_response,
                "message": f"用户认知水平评估完成，判定为{user_level_assessment['user_level']}级别"
            }
            
            logger.info(f"User level assessment completed: {user_level_assessment['user_level']} "
                       f"(confidence: {user_level_assessment['confidence']:.2%})")
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in user level adaptive response: {str(e)}", exc_info=True)
            error_msg = f"用户水平自适应响应处理失败: {str(e)}"
            raise Exception(error_msg)