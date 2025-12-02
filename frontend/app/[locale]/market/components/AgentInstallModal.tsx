"use client";

import React, { useState, useEffect } from "react";
import { Modal, Steps, Button, Select, Input, Form, Descriptions, Tag, Space, Spin, App } from "antd";
import { CheckCircleOutlined, DownloadOutlined, CloseCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { MarketAgentDetail } from "@/types/market";
import { ModelOption } from "@/types/modelConfig";
import { modelService } from "@/services/modelService";
import { getMcpServerList, addMcpServer } from "@/services/mcpService";
import { McpServer } from "@/types/agentConfig";
import { useAgentImport } from "@/hooks/useAgentImport";
import log from "@/lib/logger";

interface AgentInstallModalProps {
  visible: boolean;
  onCancel: () => void;
  agentDetails: MarketAgentDetail | null;
  onInstallComplete?: () => void;
}

interface ConfigField {
  fieldPath: string; // e.g., "duty_prompt", "tools[0].params.api_key"
  fieldLabel: string; // User-friendly label
  promptHint?: string; // Hint from <TO_CONFIG:XXXX>
  currentValue: string;
}

interface McpServerToInstall {
  mcp_server_name: string;
  mcp_url: string;
  isInstalled: boolean;
  isUrlEditable: boolean; // true if url is <TO_CONFIG>
  editedUrl?: string;
}

const needsConfig = (value: any): boolean => {
  if (typeof value === "string") {
    return value.trim() === "<TO_CONFIG>" || value.trim().startsWith("<TO_CONFIG:");
  }
  return false;
};

const extractPromptHint = (value: string): string | undefined => {
  if (typeof value !== "string") return undefined;
  const match = value.trim().match(/^<TO_CONFIG:(.+)>$/);
  return match ? match[1] : undefined;
};

export default function AgentInstallModal({
  visible,
  onCancel,
  agentDetails,
  onInstallComplete,
}: AgentInstallModalProps) {
  const { t, i18n } = useTranslation("common");
  const isZh = i18n.language === "zh" || i18n.language === "zh-CN";
  const { message } = App.useApp();

  // Use unified import hook
  const { importFromData, isImporting: isInstallingAgent } = useAgentImport({
    onSuccess: () => {
      onInstallComplete?.();
    },
    onError: (error) => {
      message.error(error.message || t("market.install.error.installFailed", "Failed to install agent"));
    },
  });

  const [currentStep, setCurrentStep] = useState(0);
  const [llmModels, setLlmModels] = useState<ModelOption[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string>("");

  const [configFields, setConfigFields] = useState<ConfigField[]>([]);
  const [configValues, setConfigValues] = useState<Record<string, string>>({});

  const [mcpServers, setMcpServers] = useState<McpServerToInstall[]>([]);
  const [existingMcpServers, setExistingMcpServers] = useState<McpServer[]>([]);
  const [loadingMcpServers, setLoadingMcpServers] = useState(false);
  const [installingMcp, setInstallingMcp] = useState<Record<string, boolean>>({});

  // Load LLM models
  useEffect(() => {
    if (visible) {
      loadLLMModels();
    }
  }, [visible]);

  // Parse agent details for config fields and MCP servers
  useEffect(() => {
    if (visible && agentDetails) {
      parseConfigFields();
      parseMcpServers();
    }
  }, [visible, agentDetails]);

  const loadLLMModels = async () => {
    setLoadingModels(true);
    try {
      const models = await modelService.getLLMModels();
      setLlmModels(models.filter(m => m.connect_status === "available"));
      
      // Auto-select first available model
      if (models.length > 0 && models[0].connect_status === "available") {
        setSelectedModelId(models[0].id);
        setSelectedModelName(models[0].displayName);
      }
    } catch (error) {
      log.error("Failed to load LLM models:", error);
      message.error(t("market.install.error.loadModels", "Failed to load models"));
    } finally {
      setLoadingModels(false);
    }
  };

  const parseConfigFields = () => {
    if (!agentDetails) return;

    const fields: ConfigField[] = [];

    // Check basic fields (excluding MCP-related fields)
    const basicFields: Array<{ key: keyof MarketAgentDetail; label: string }> = [
      { key: "description", label: t("market.detail.description", "Description") },
      { key: "business_description", label: t("market.detail.businessDescription", "Business Description") },
      { key: "duty_prompt", label: t("market.detail.dutyPrompt", "Duty Prompt") },
      { key: "constraint_prompt", label: t("market.detail.constraintPrompt", "Constraint Prompt") },
      { key: "few_shots_prompt", label: t("market.detail.fewShotsPrompt", "Few Shots Prompt") },
    ];

    basicFields.forEach(({ key, label }) => {
      const value = agentDetails[key];
      if (needsConfig(value)) {
        fields.push({
          fieldPath: key,
          fieldLabel: label,
          promptHint: extractPromptHint(value as string),
          currentValue: value as string,
        });
      }
    });

    // Check tool params (excluding MCP server names/urls)
    agentDetails.tools?.forEach((tool, toolIndex) => {
      if (tool.params && typeof tool.params === "object") {
        Object.entries(tool.params).forEach(([paramKey, paramValue]) => {
          if (needsConfig(paramValue)) {
            fields.push({
              fieldPath: `tools[${toolIndex}].params.${paramKey}`,
              fieldLabel: `${tool.name || tool.class_name} - ${paramKey}`,
              promptHint: extractPromptHint(paramValue as string),
              currentValue: paramValue as string,
            });
          }
        });
      }
    });

    setConfigFields(fields);

    // Initialize config values
    const initialValues: Record<string, string> = {};
    fields.forEach(field => {
      initialValues[field.fieldPath] = "";
    });
    setConfigValues(initialValues);
  };

  const parseMcpServers = async () => {
    if (!agentDetails?.mcp_servers || agentDetails.mcp_servers.length === 0) {
      setMcpServers([]);
      return;
    }

    setLoadingMcpServers(true);
    try {
      // Load existing MCP servers from system
      const result = await getMcpServerList();
      const existing = result.success ? result.data : [];
      setExistingMcpServers(existing);

      // Check each MCP server
      const serversToInstall: McpServerToInstall[] = agentDetails.mcp_servers.map(mcp => {
        const isUrlConfigNeeded = needsConfig(mcp.mcp_url);
        
        // Check if already installed (match by both name and url)
        const isInstalled = !isUrlConfigNeeded && existing.some(
          (existingMcp: McpServer) => 
            existingMcp.service_name === mcp.mcp_server_name && 
            existingMcp.mcp_url === mcp.mcp_url
        );

        return {
          mcp_server_name: mcp.mcp_server_name,
          mcp_url: mcp.mcp_url,
          isInstalled,
          isUrlEditable: isUrlConfigNeeded,
          editedUrl: isUrlConfigNeeded ? "" : mcp.mcp_url,
        };
      });

      setMcpServers(serversToInstall);
    } catch (error) {
      log.error("Failed to check MCP servers:", error);
      message.error(t("market.install.error.checkMcp", "Failed to check MCP servers"));
    } finally {
      setLoadingMcpServers(false);
    }
  };

  const handleMcpUrlChange = (index: number, newUrl: string) => {
    setMcpServers(prev => {
      const updated = [...prev];
      updated[index].editedUrl = newUrl;
      return updated;
    });
  };

  const handleInstallMcp = async (index: number) => {
    const mcp = mcpServers[index];
    const urlToUse = mcp.editedUrl || mcp.mcp_url;

    if (!urlToUse || urlToUse.trim() === "") {
      message.error(t("market.install.error.mcpUrlRequired", "MCP URL is required"));
      return;
    }

    const key = `${index}`;
    setInstallingMcp(prev => ({ ...prev, [key]: true }));

    try {
      const result = await addMcpServer(urlToUse, mcp.mcp_server_name);
      if (result.success) {
        message.success(t("market.install.success.mcpInstalled", "MCP server installed successfully"));
        // Mark as installed - update state directly without re-fetching
        setMcpServers(prev => {
          const updated = [...prev];
          updated[index].isInstalled = true;
          updated[index].editedUrl = urlToUse;
          return updated;
        });
      } else {
        message.error(result.message || t("market.install.error.mcpInstall", "Failed to install MCP server"));
      }
    } catch (error) {
      log.error("Failed to install MCP server:", error);
      message.error(t("market.install.error.mcpInstall", "Failed to install MCP server"));
    } finally {
      setInstallingMcp(prev => ({ ...prev, [key]: false }));
    }
  };

  const handleNext = () => {
    if (currentStep === 0) {
      // Step 1: Model selection validation
      if (!selectedModelId || !selectedModelName) {
        message.error(t("market.install.error.modelRequired", "Please select a model"));
        return;
      }
    } else if (currentStep === 1) {
      // Step 2: Config fields validation
      const emptyFields = configFields.filter(field => !configValues[field.fieldPath]?.trim());
      if (emptyFields.length > 0) {
        message.error(t("market.install.error.configRequired", "Please fill in all required fields"));
        return;
      }
    }

    setCurrentStep(prev => prev + 1);
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleInstall = async () => {
    try {
      // Prepare the data structure for import
      const importData = prepareImportData();
      
      if (!importData) {
        message.error(t("market.install.error.invalidData", "Invalid agent data"));
        return;
      }

      log.info("Importing agent with data:", importData);

      // Import using unified hook
      await importFromData(importData);
      
      // Success message will be shown by onSuccess callback
      message.success(t("market.install.success", "Agent installed successfully!"));
    } catch (error) {
      // Error message will be shown by onError callback
      log.error("Failed to install agent:", error);
    }
  };

  const prepareImportData = () => {
    if (!agentDetails) return null;

    // Clone agent_json structure
    const agentJson = JSON.parse(JSON.stringify(agentDetails.agent_json));

    // Update model information
    const agentInfo = agentJson.agent_info[String(agentDetails.agent_id)];
    if (agentInfo) {
      agentInfo.model_id = selectedModelId;
      agentInfo.model_name = selectedModelName;
      
      // Clear business logic model fields
      agentInfo.business_logic_model_id = null;
      agentInfo.business_logic_model_name = null;

      // Update config fields
      configFields.forEach(field => {
        const value = configValues[field.fieldPath];
        if (field.fieldPath.includes("tools[")) {
          // Handle tool params
          const match = field.fieldPath.match(/tools\[(\d+)\]\.params\.(.+)/);
          if (match && agentInfo.tools) {
            const toolIndex = parseInt(match[1]);
            const paramKey = match[2];
            if (agentInfo.tools[toolIndex]) {
              agentInfo.tools[toolIndex].params[paramKey] = value;
            }
          }
        } else {
          // Handle basic fields
          agentInfo[field.fieldPath] = value;
        }
      });

      // Update MCP info
      if (agentJson.mcp_info) {
        agentJson.mcp_info = agentJson.mcp_info.map((mcp: any) => {
          const matchingServer = mcpServers.find(
            s => s.mcp_server_name === mcp.mcp_server_name
          );
          if (matchingServer && matchingServer.editedUrl) {
            return {
              ...mcp,
              mcp_url: matchingServer.editedUrl,
            };
          }
          return mcp;
        });
      }
    }

    return agentJson;
  };

  const handleCancel = () => {
    // Reset state
    setCurrentStep(0);
    setSelectedModelId(null);
    setSelectedModelName("");
    setConfigFields([]);
    setConfigValues({});
    setMcpServers([]);
    onCancel();
  };

  // Filter only required steps for navigation
  const steps = [
    {
      key: "model",
      title: t("market.install.step.model", "Select Model"),
    },
    configFields.length > 0 && {
      key: "config",
      title: t("market.install.step.config", "Configure Fields"),
    },
    mcpServers.length > 0 && {
      key: "mcp",
      title: t("market.install.step.mcp", "MCP Servers"),
    },
  ].filter(Boolean) as Array<{ key: string; title: string }>;

  // Check if can proceed to next step
  const canProceed = () => {
    const currentStepKey = steps[currentStep]?.key;
    
    if (currentStepKey === "model") {
      return selectedModelId !== null && selectedModelName !== "";
    } else if (currentStepKey === "config") {
      return configFields.every(field => configValues[field.fieldPath]?.trim());
    } else if (currentStepKey === "mcp") {
      // All non-editable MCPs should be installed or have edited URLs
      return mcpServers.every(mcp => 
        mcp.isInstalled || 
        (mcp.isUrlEditable && mcp.editedUrl && mcp.editedUrl.trim() !== "") ||
        (!mcp.isUrlEditable && mcp.mcp_url && mcp.mcp_url.trim() !== "")
      );
    }
    
    return true;
  };

  const renderStepContent = () => {
    const currentStepKey = steps[currentStep]?.key;

    if (currentStepKey === "model") {
      return (
        <div className="space-y-6">
          {/* Agent Info - Title and Description Style */}
          {agentDetails && (
            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg p-6 border border-purple-100 dark:border-purple-800">
              <h3 className="text-xl font-bold text-purple-900 dark:text-purple-100 mb-2">
                {agentDetails.display_name}
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                {agentDetails.description}
              </p>
            </div>
          )}

          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {t("market.install.model.description", "Select a model from your configured models to use for this agent.")}
            </p>
            
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
                {t("market.install.model.label", "Model")}
                <span className="text-red-500 ml-1">*</span>
              </label>
              <div className="flex-1">
                {loadingModels ? (
                  <Spin />
                ) : (
                  <Select
                    value={selectedModelName || undefined}
                    onChange={(value, option) => {
                      const modelId = option && 'key' in option ? Number(option.key) : null;
                      setSelectedModelName(value);
                      setSelectedModelId(modelId);
                    }}
                    size="large"
                    style={{ width: "100%" }}
                    placeholder={t("market.install.model.placeholder", "Select a model")}
                  >
                    {llmModels.map((model) => (
                      <Select.Option key={model.id} value={model.displayName}>
                        {model.displayName}
                      </Select.Option>
                    ))}
                  </Select>
                )}
              </div>
            </div>

            {llmModels.length === 0 && !loadingModels && (
              <div className="text-sm text-red-600 mt-2">
                {t("market.install.model.noModels", "No available models. Please configure models first.")}
              </div>
            )}
          </div>
        </div>
      );
    } else if (currentStepKey === "config") {
      return (
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            {t("market.install.config.description", "Please configure the following required fields for this agent.")}
          </p>

          <Form layout="vertical">
            {configFields.map((field) => (
              <Form.Item
                key={field.fieldPath}
                label={
                  <span>
                    {field.fieldLabel}
                    <span className="text-red-500 ml-1">*</span>
                  </span>
                }
                required={false}
              >
                <Input.TextArea
                  value={configValues[field.fieldPath] || ""}
                  onChange={(e) => {
                    setConfigValues(prev => ({
                      ...prev,
                      [field.fieldPath]: e.target.value,
                    }));
                  }}
                  placeholder={field.promptHint || t("market.install.config.placeholder", "Enter configuration value")}
                  rows={3}
                  size="large"
                />
              </Form.Item>
            ))}
          </Form>
        </div>
      );
    } else if (currentStepKey === "mcp") {
      return (
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            {t("market.install.mcp.description", "This agent requires the following MCP servers. Please install or configure them.")}
          </p>

          {loadingMcpServers ? (
            <div className="text-center py-8">
              <Spin />
            </div>
          ) : (
            <div className="space-y-3">
              {mcpServers.map((mcp, index) => (
                <div
                  key={`${mcp.mcp_server_name}-${index}`}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 min-h-[120px] flex items-center"
                >
                  <div className="flex items-center justify-between w-full gap-4">
                    <div className="flex-1 flex flex-col justify-center">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="font-medium text-base">
                          {mcp.mcp_server_name}
                        </span>
                        {mcp.isInstalled ? (
                          <Tag icon={<CheckCircleOutlined />} color="success" className="text-sm">
                            {t("market.install.mcp.installed", "Installed")}
                          </Tag>
                        ) : (
                          <Tag icon={<CloseCircleOutlined />} color="default" className="text-sm">
                            {t("market.install.mcp.notInstalled", "Not Installed")}
                          </Tag>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
                          MCP URL:
                        </span>
                        {(mcp.isUrlEditable || !mcp.isInstalled) ? (
                          <Input
                            value={mcp.editedUrl || ""}
                            onChange={(e) => handleMcpUrlChange(index, e.target.value)}
                            placeholder={mcp.isUrlEditable 
                              ? t("market.install.mcp.urlPlaceholder", "Enter MCP server URL")
                              : mcp.mcp_url
                            }
                            size="middle"
                            disabled={mcp.isInstalled}
                            style={{ maxWidth: "400px" }}
                          />
                        ) : (
                          <span className="text-sm text-gray-700 dark:text-gray-300 break-all">
                            {mcp.editedUrl || mcp.mcp_url}
                          </span>
                        )}
                      </div>
                    </div>

                    {!mcp.isInstalled && (
                      <Button
                        type="primary"
                        size="middle"
                        icon={<PlusOutlined />}
                        onClick={() => handleInstallMcp(index)}
                        loading={installingMcp[String(index)]}
                        disabled={!mcp.editedUrl || mcp.editedUrl.trim() === ""}
                        className="flex-shrink-0"
                      >
                        {t("market.install.mcp.install", "Install")}
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  const isLastStep = currentStep === steps.length - 1;

  return (
    <Modal
      title={
        <div className="flex items-center gap-2">
          <DownloadOutlined />
          <span>{t("market.install.title", "Install Agent")}</span>
        </div>
      }
      open={visible}
      onCancel={handleCancel}
      width={800}
      footer={
        <div className="flex justify-between">
          <Button onClick={handleCancel}>
            {t("common.cancel", "Cancel")}
          </Button>
          <Space>
            {currentStep > 0 && (
              <Button onClick={handlePrevious}>
                {t("market.install.button.previous", "Previous")}
              </Button>
            )}
            {!isLastStep && (
              <Button
                type="primary"
                onClick={handleNext}
                disabled={!canProceed()}
              >
                {t("market.install.button.next", "Next")}
              </Button>
            )}
            {isLastStep && (
              <Button
                type="primary"
                onClick={handleInstall}
                disabled={!canProceed()}
                loading={isInstallingAgent}
                icon={<DownloadOutlined />}
              >
                {isInstallingAgent 
                  ? t("market.install.button.installing", "Installing...")
                  : t("market.install.button.install", "Install")}
              </Button>
            )}
          </Space>
        </div>
      }
    >
      <div className="py-4">
        <Steps
          current={currentStep}
          items={steps.map(step => ({
            title: step.title,
          }))}
          className="mb-6"
        />

        <div className="min-h-[300px]">
          {renderStepContent()}
        </div>
      </div>
    </Modal>
  );
}

