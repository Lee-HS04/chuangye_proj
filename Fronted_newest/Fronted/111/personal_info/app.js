// 当前步骤
let currentStep = 1;
const totalSteps = 3;

// 步骤切换函数
function nextStep() {
    if (currentStep < totalSteps) {
        // 验证当前步骤的表单
        if (validateStep(currentStep)) {
            document.getElementById(`step${currentStep}`).classList.remove('active');
            currentStep++;
            document.getElementById(`step${currentStep}`).classList.add('active');
            updateProgress();
            updateStepIndicators();
        }
    }
}

function prevStep() {
    if (currentStep > 1) {
        document.getElementById(`step${currentStep}`).classList.remove('active');
        currentStep--;
        document.getElementById(`step${currentStep}`).classList.add('active');
        updateProgress();
        updateStepIndicators();
    }
}

// 更新进度条
function updateProgress() {
    const progressPercentage = (currentStep / totalSteps) * 100;
    document.getElementById('progressFill').style.width = `${progressPercentage}%`;
}

// 更新步骤指示器
function updateStepIndicators() {
    for (let i = 1; i <= totalSteps; i++) {
        const stepIndicator = document.querySelector(`.step-indicator[data-step="${i}"]`);
        if (i < currentStep) {
            stepIndicator.classList.add('completed');
            stepIndicator.classList.remove('active');
        } else if (i === currentStep) {
            stepIndicator.classList.add('active');
            stepIndicator.classList.remove('completed');
        } else {
            stepIndicator.classList.remove('active', 'completed');
        }
    }
}

// 表单验证
function validateStep(step) {
    switch (step) {
        case 1:
            // 验证基础信息
            const age = document.getElementById('age').value;
            const gender = document.querySelector('input[name="gender"]:checked');
            const height = document.getElementById('height').value;
            const weight = document.getElementById('weight').value;
            const experience = document.querySelector('input[name="experience"]:checked');
            const goals = document.querySelectorAll('input[name="goal"]:checked');
            
            if (!age || age < 10 || age > 100) {
                alert('请输入有效的年龄（10-100岁）');
                return false;
            }
            
            if (!gender) {
                alert('请选择性别');
                return false;
            }
            
            if (!height || height < 100 || height > 250) {
                alert('请输入有效的身高（100-250cm）');
                return false;
            }
            
            if (!weight || weight < 30 || weight > 200) {
                alert('请输入有效的体重（30-200kg）');
                return false;
            }
            
            if (!experience) {
                alert('请选择运动经验');
                return false;
            }
            
            if (goals.length === 0) {
                alert('请至少选择一个运动目标');
                return false;
            }
            
            return true;
            
        case 2:
            // 验证生活状态
            const work = document.querySelector('input[name="work"]:checked');
            const frequency = document.querySelector('input[name="frequency"]:checked');
            const sleep = document.querySelector('input[name="sleep"]:checked');
            
            if (!work) {
                alert('请选择工作状态');
                return false;
            }
            
            if (!frequency) {
                alert('请选择每周运动频率');
                return false;
            }
            
            if (!sleep) {
                alert('请选择睡眠质量');
                return false;
            }
            
            return true;
            
        case 3:
            // 验证健康状况
            const hasInjury = document.querySelector('input[name="hasInjury"]:checked');
            const hasDiscomfort = document.querySelector('input[name="hasDiscomfort"]:checked');
            
            if (!hasInjury) {
                alert('请选择是否有旧伤');
                return false;
            }
            
            if (!hasDiscomfort) {
                alert('请选择是否有不适');
                return false;
            }
            
            // 如果有旧伤，验证受伤部位
            if (hasInjury.value === 'yes') {
                const injuryParts = document.querySelectorAll('input[name="injuryPart"]:checked');
                const injuryTypes = document.querySelectorAll('input[name="injuryType"]:checked');
                
                if (injuryParts.length === 0) {
                    alert('请选择受伤部位');
                    return false;
                }
                
                if (injuryTypes.length === 0) {
                    alert('请选择旧伤类型');
                    return false;
                }
            }
            
            // 如果有不适，验证不适部位
            if (hasDiscomfort.value === 'yes') {
                const discomfortParts = document.querySelectorAll('input[name="discomfortPart"]:checked');
                const symptoms = document.querySelectorAll('input[name="symptom"]:checked');
                
                if (discomfortParts.length === 0) {
                    alert('请选择不适部位');
                    return false;
                }
                
                if (symptoms.length === 0) {
                    alert('请选择症状描述');
                    return false;
                }
            }
            
            return true;
            
        default:
            return true;
    }
}

// 切换旧伤详情部分
function toggleInjurySection(show) {
    const injuryDetails = document.getElementById('injuryDetails');
    if (show) {
        injuryDetails.style.display = 'block';
    } else {
        injuryDetails.style.display = 'none';
        // 重置表单
        document.querySelectorAll('input[name="injuryPart"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        document.querySelectorAll('input[name="injuryType"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    }
}

// 切换不适详情部分
function toggleDiscomfortSection(show) {
    const discomfortDetails = document.getElementById('discomfortDetails');
    if (show) {
        discomfortDetails.style.display = 'block';
    } else {
        discomfortDetails.style.display = 'none';
        // 重置表单
        document.querySelectorAll('input[name="discomfortPart"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        document.querySelectorAll('input[name="symptom"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    }
}

// 收集表单数据
function collectFormData() {
    // 基础信息
    const basicInfo = {
        age: document.getElementById('age').value,
        gender: document.querySelector('input[name="gender"]:checked')?.value,
        height: document.getElementById('height').value,
        weight: document.getElementById('weight').value,
        experience: document.querySelector('input[name="experience"]:checked')?.value,
        goals: Array.from(document.querySelectorAll('input[name="goal"]:checked')).map(cb => cb.value)
    };
    
    // 生活状态
    const lifeStatus = {
        work: document.querySelector('input[name="work"]:checked')?.value,
        frequency: document.querySelector('input[name="frequency"]:checked')?.value,
        activities: Array.from(document.querySelectorAll('input[name="activity"]:checked')).map(cb => cb.value),
        sleep: document.querySelector('input[name="sleep"]:checked')?.value
    };
    
    // 健康状况
    const healthStatus = {
        hasInjury: document.querySelector('input[name="hasInjury"]:checked')?.value,
        injuryParts: Array.from(document.querySelectorAll('input[name="injuryPart"]:checked')).map(cb => cb.value),
        injuryTypes: Array.from(document.querySelectorAll('input[name="injuryType"]:checked')).map(cb => cb.value),
        hasDiscomfort: document.querySelector('input[name="hasDiscomfort"]:checked')?.value,
        discomfortParts: Array.from(document.querySelectorAll('input[name="discomfortPart"]:checked')).map(cb => cb.value),
        symptoms: Array.from(document.querySelectorAll('input[name="symptom"]:checked')).map(cb => cb.value),
        otherHealth: document.getElementById('otherHealth').value
    };
    
    return {
        basicInfo,
        lifeStatus,
        healthStatus,
        updatedAt: new Date().toISOString()
    };
}

// 同步数据到JSON文件
function syncUserDataToJsonFile(userData) {
    return fetch('/api/update_user_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(userData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('用户数据已成功同步到JSON文件');
        } else {
            console.error('同步JSON文件失败:', data.message);
        }
        return data;
    })
    .catch(error => {
        console.error('同步JSON文件时发生错误:', error);
        return { success: false, message: error.message };
    });
}

// 提交表单
async function submitForm() {
    if (validateStep(currentStep)) {
        const formData = collectFormData();
        
        // 获取当前用户信息
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        if (currentUser) {
            // 更新当前用户的个人信息
            currentUser.personal_info = formData;
            currentUser.updated_at = new Date().toISOString();

            // 更新localStorage中的用户数据
            const userData = JSON.parse(localStorage.getItem('userData') || '{}');
            if (userData.users) {
                const userIndex = userData.users.findIndex(u => u.id === currentUser.id);
                if (userIndex !== -1) {
                    userData.users[userIndex] = currentUser;
                } else {
                    userData.users.push(currentUser);
                }
                userData.last_updated = new Date().toISOString();
                localStorage.setItem('userData', JSON.stringify(userData));

                // 同步数据到JSON文件
                await syncUserDataToJsonFile(userData);
            }

            // 更新当前用户信息
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
        }
        
        // 显示完成页面
        document.getElementById(`step${currentStep}`).classList.remove('active');
        document.getElementById('step4').classList.add('active');
        
        // 3秒后跳转到主页
        setTimeout(() => {
            window.location.href = '../main_features/main.html';
        }, 3000);
    }
}

// 页面加载时的初始化
window.addEventListener('DOMContentLoaded', function() {
    // 初始化进度条和步骤指示器
    updateProgress();
    updateStepIndicators();
    
    // 预填个人信息
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    if (currentUser && currentUser.personal_info) {
        const personalInfo = currentUser.personal_info;
        
        // 预填基础信息
        if (personalInfo.basicInfo) {
            if (personalInfo.basicInfo.age) {
                document.getElementById('age').value = personalInfo.basicInfo.age;
            }
            if (personalInfo.basicInfo.gender) {
                const genderRadio = document.querySelector(`input[name="gender"][value="${personalInfo.basicInfo.gender}"]`);
                if (genderRadio) genderRadio.checked = true;
            }
            if (personalInfo.basicInfo.height) {
                document.getElementById('height').value = personalInfo.basicInfo.height;
            }
            if (personalInfo.basicInfo.weight) {
                document.getElementById('weight').value = personalInfo.basicInfo.weight;
            }
            if (personalInfo.basicInfo.experience) {
                const experienceRadio = document.querySelector(`input[name="experience"][value="${personalInfo.basicInfo.experience}"]`);
                if (experienceRadio) experienceRadio.checked = true;
            }
            if (personalInfo.basicInfo.goals) {
                personalInfo.basicInfo.goals.forEach(goal => {
                    const goalCheckbox = document.querySelector(`input[name="goal"][value="${goal}"]`);
                    if (goalCheckbox) goalCheckbox.checked = true;
                });
            }
        }
        
        // 预填生活状态
        if (personalInfo.lifeStatus) {
            if (personalInfo.lifeStatus.work) {
                const workRadio = document.querySelector(`input[name="work"][value="${personalInfo.lifeStatus.work}"]`);
                if (workRadio) workRadio.checked = true;
            }
            if (personalInfo.lifeStatus.frequency) {
                const frequencyRadio = document.querySelector(`input[name="frequency"][value="${personalInfo.lifeStatus.frequency}"]`);
                if (frequencyRadio) frequencyRadio.checked = true;
            }
            if (personalInfo.lifeStatus.activities) {
                personalInfo.lifeStatus.activities.forEach(activity => {
                    const activityCheckbox = document.querySelector(`input[name="activity"][value="${activity}"]`);
                    if (activityCheckbox) activityCheckbox.checked = true;
                });
            }
            if (personalInfo.lifeStatus.sleep) {
                const sleepRadio = document.querySelector(`input[name="sleep"][value="${personalInfo.lifeStatus.sleep}"]`);
                if (sleepRadio) sleepRadio.checked = true;
            }
        }
        
        // 预填健康状况
        if (personalInfo.healthStatus) {
            if (personalInfo.healthStatus.hasInjury) {
                const injuryRadio = document.querySelector(`input[name="hasInjury"][value="${personalInfo.healthStatus.hasInjury}"]`);
                if (injuryRadio) injuryRadio.checked = true;
                
                // 处理旧伤详情
                if (personalInfo.healthStatus.hasInjury === 'yes') {
                    toggleInjurySection(true);
                    if (personalInfo.healthStatus.injuryParts) {
                        personalInfo.healthStatus.injuryParts.forEach(part => {
                            const partCheckbox = document.querySelector(`input[name="injuryPart"][value="${part}"]`);
                            if (partCheckbox) partCheckbox.checked = true;
                        });
                    }
                    if (personalInfo.healthStatus.injuryTypes) {
                        personalInfo.healthStatus.injuryTypes.forEach(type => {
                            const typeCheckbox = document.querySelector(`input[name="injuryType"][value="${type}"]`);
                            if (typeCheckbox) typeCheckbox.checked = true;
                        });
                    }
                }
            }
            
            if (personalInfo.healthStatus.hasDiscomfort) {
                const discomfortRadio = document.querySelector(`input[name="hasDiscomfort"][value="${personalInfo.healthStatus.hasDiscomfort}"]`);
                if (discomfortRadio) discomfortRadio.checked = true;
                
                // 处理不适详情
                if (personalInfo.healthStatus.hasDiscomfort === 'yes') {
                    toggleDiscomfortSection(true);
                    if (personalInfo.healthStatus.discomfortParts) {
                        personalInfo.healthStatus.discomfortParts.forEach(part => {
                            const partCheckbox = document.querySelector(`input[name="discomfortPart"][value="${part}"]`);
                            if (partCheckbox) partCheckbox.checked = true;
                        });
                    }
                    if (personalInfo.healthStatus.symptoms) {
                        personalInfo.healthStatus.symptoms.forEach(symptom => {
                            const symptomCheckbox = document.querySelector(`input[name="symptom"][value="${symptom}"]`);
                            if (symptomCheckbox) symptomCheckbox.checked = true;
                        });
                    }
                }
            }
            
            if (personalInfo.healthStatus.otherHealth) {
                document.getElementById('otherHealth').value = personalInfo.healthStatus.otherHealth;
            }
        }
    }
});