$(function() {
    $( "#tabs" ).tabs();
});
$(function() {
    $( "#tabs_inter" ).tabs({
        collapsible: true
    });
});
$(function() {
    $( document ).tooltip({
        track: true, hide: {duration: 0}
        ,position: { my: "left-20 top+25", at: "right bottom" }
    });
});
$(document).ready(function(){ // 必须有这一行，在页面加载之后执行，否则无效（不加的时候真的无效，已尝试）。
    $('a.pop0').unbind("click").click(function(){ //.unbind("click") 部分解决（点词条重复触发的问题解决，但重新查询后重新触发还存在）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
        $('a.pop0').popover({ trigger: "manual" , html: true, animation:false})
            .on("mouseover", function () {
                var _this = this;
                $(this).unbind("click").click(function () { //.unbind("click") 部分解决（同上）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
                    $(this).popover("show");
                    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 记录点击结果。
                    var description = $("#description").val();
                    //console.log($(_this).text()+"||"+description);
                    $.get("/feedback/", { 'content': $(_this).text()+"||"+description, 'mode': 'FBW' });
                    $(".popover").on("mouseleave", function () { //【解决不易，这个很重要】
                        $(_this).popover('hide'); 
                    });
                });
            }).on("mouseout", function () { //mouseleave也有问题，在弹框里出现tip时，指针移到tip上就相当于离开目标了，此时弹框会消失（按需求是不应该消失的）
                var _this = this;
                setTimeout(function () {
                    if (!$(".popover:hover").length) {
                        $(_this).popover("hide");
                        if ($(window).width()>768) { //手机端不能加这一条，会发生框只闪一下而不显示的问题。【解决不易，这个很重要】
                            $("div.popover").hide(); //清理卡死的popover弹框
                        }
                    }
                }, 200);
            });
    });
    $('a.pop1').unbind("click").click(function(){ //.unbind("click") 部分解决（点词条重复触发的问题解决，但重新查询后重新触发还存在）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
        $('a.pop1').popover({ trigger: "manual" , html: true, animation:false})
            .on("mouseover", function () {
                var _this = this;
                $(this).unbind("click").click(function () { //.unbind("click") 部分解决（同上）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
                    $(this).popover("show");
                    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 记录点击结果。
                    var description = $("#description_EE").val();
                    //console.log($(_this).text()+"||"+description);
                    $.get("/feedback/", { 'content': $(_this).text()+"||"+description, 'mode': 'FBW' });
                    $(".popover").on("mouseleave", function () { //【解决不易，这个很重要】
                        $(_this).popover('hide'); 
                    });
                });
            }).on("mouseout", function () { //mouseleave也有问题，在弹框里出现tip时，指针移到tip上就相当于离开目标了，此时弹框会消失（按需求是不应该消失的）
                var _this = this;
                setTimeout(function () {
                    if (!$(".popover:hover").length) {
                        $(_this).popover("hide");
                        if ($(window).width()>768) { //手机端不能加这一条，会发生框只闪一下而不显示的问题。【解决不易，这个很重要】
                            $("div.popover").hide(); //清理卡死的popover弹框
                        }
                    }
                }, 200);
            });
    });
    $('a.pop2').unbind("click").click(function(){ //.unbind("click") 部分解决（点词条重复触发的问题解决，但重新查询后重新触发还存在）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
        $('a.pop2').popover({ trigger: "manual" , html: true, animation:false})
            .on("mouseover", function () {
                var _this = this;
                $(this).unbind("click").click(function () { //.unbind("click") 部分解决（同上）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
                    $(this).popover("show");
                    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 记录点击结果。
                    var description = $("#description_CE").val();
                    //console.log($(_this).text()+"||"+description);
                    $.get("/feedback/", { 'content': $(_this).text()+"||"+description, 'mode': 'FBW' });
                    $(".popover").on("mouseleave", function () { //【解决不易，这个很重要】
                        $(_this).popover('hide'); 
                    });
                });
            }).on("mouseout", function () { //mouseleave也有问题，在弹框里出现tip时，指针移到tip上就相当于离开目标了，此时弹框会消失（按需求是不应该消失的）
                var _this = this;
                setTimeout(function () {
                    if (!$(".popover:hover").length) {
                        $(_this).popover("hide");
                        if ($(window).width()>768) { //手机端不能加这一条，会发生框只闪一下而不显示的问题。【解决不易，这个很重要】
                            $("div.popover").hide(); //清理卡死的popover弹框
                        }
                    }
                }, 200);
            });
    });
    $('a.pop3').unbind("click").click(function(){ //.unbind("click") 部分解决（点词条重复触发的问题解决，但重新查询后重新触发还存在）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
        $('a.pop3').popover({ trigger: "manual" , html: true, animation:false})
            .on("mouseover", function () {
                var _this = this;
                $(this).unbind("click").click(function () { //.unbind("click") 部分解决（同上）重复绑定click从而重复触发click事件的问题【解决不易，这个很重要】
                    $(this).popover("show");
                    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 记录点击结果。
                    var description = $("#description_EC").val();
                    //console.log($(_this).text()+"||"+description);
                    $.get("/feedback/", { 'content': $(_this).text()+"||"+description, 'mode': 'FBW' });
                    $(".popover").on("mouseleave", function () { //【解决不易，这个很重要】
                        $(_this).popover('hide'); 
                    });
                });
            }).on("mouseout", function () { //mouseleave也有问题，在弹框里出现tip时，指针移到tip上就相当于离开目标了，此时弹框会消失（按需求是不应该消失的）
                var _this = this;
                setTimeout(function () {
                    if (!$(".popover:hover").length) {
                        $(_this).popover("hide");
                        if ($(window).width()>768) { //手机端不能加这一条，会发生框只闪一下而不显示的问题。【解决不易，这个很重要】
                            $("div.popover").hide(); //清理卡死的popover弹框
                        }
                    }
                }, 200);
            });
    });
});

//<!--反馈信息的获取-->
var getSelectedTabId = 0;
$(function () { 
    $('#tabs').tabs({
        activate: function (event, ui) {
            var activeTab = $('#tabs').tabs('option', 'active');
            getSelectedTabId = activeTab;
            // 适配英文界面
            if (getSelectedTabId%2==1) {
                $("#id_clk1").attr("value","Propose Appropriate Words");
                $("#id_clk2").attr("value","Make Suggestions");
                $("#id_home").text("Home Page");
                $("#id_about").attr("href","../about_en/").text("About Us");
                $("#id_link").text("GitHub Link");
                $("#idm_home").html('<span class="glyphicon glyphicon-home"></span> Home');
                $("#idm_about").attr("href","../about_en/").text("About Us");
                $("#idm_link").text("GitHub Link");
            }
            else {
                $("#id_clk1").attr("value","点此反馈推荐词");
                $("#id_clk2").attr("value","点此反馈意见建议");
                $("#id_home").text("反向词典主页");
                $("#id_about").attr("href","../about/").text("关于我们");
                $("#id_link").text("GitHub链接");
                $("#idm_home").html('<span class="glyphicon glyphicon-home"></span> 主页');
                $("#idm_about").attr("href","../about/").text("关于我们");
                $("#idm_link").text("GitHub链接");
            };
        }
    });
})

function diagWord() {
    if (getSelectedTabId%2==0) {
        var str=prompt("未能帮您找到想要的词？\n您认为与您描述相近的词:（多词可用标点分隔）","");
    }
    else {
        var str=prompt("Any appropriate words in your opinion:","");
    };
    if (str!="" && str!=null) {
        if (getSelectedTabId%2==0) {
            alert("谢谢您的指导，本站希望与您一起学习，共同提升自然语言理解和应用能力！");
        }
        else {
            alert("Thank you for your advice!");
        };
        if (getSelectedTabId==0) {
            var description = $("#description").val();
        }
        else if (getSelectedTabId==1) {
            var description = $("#description_EE").val();
        }
        else if (getSelectedTabId==2) {
            var description = $("#description_CE").val();
        }
        else if (getSelectedTabId==3) {
            var description = $("#description_EC").val();
        }
        else {
            var description = "unkown tabs";
        };
        $.get("/feedback/", { 'content': str+"|||"+description, 'mode': 'FBW' });
    }
}
function diagSuggest() {
    if (getSelectedTabId%2==0) {
        var str=prompt("您对网站有何意见或建议？","");
        if(str!="" && str!=null)
        {
            alert("感谢您的反馈！");
            $.get("/feedback/", { 'content': str, 'mode': 'FBS' });
        }
    }
    else {
        var str=prompt("Any suggestions about this website?","");
        if(str!="" && str!=null)
        {
            alert("Thanks for your feedback!");
            $.get("/feedback/", { 'content': str, 'mode': 'FBS' });
        }
    };
}
function diagError(i) {
    if (getSelectedTabId==0) {
        var word = $("#tabs-1 #li"+i).text();
    }
    else if (getSelectedTabId==1) {
        var word = $("#tabs-2 #li"+i).text();
    }
    else if (getSelectedTabId==2) {
        var word = $("#tabs-3 #li"+i).text();
    }
    else if (getSelectedTabId==3) {
        var word = $("#tabs-4 #li"+i).text();
    }
    else {
        var word = "unkown word";
    };
    if (getSelectedTabId%2==0) {
        var str=prompt("关于词“"+word+"”的相关错误描述：","");
    }
    else {
        var str=prompt('Please describe mistakes about the word "'+word+'":',"");
    };
    if (str!="" && str!=null) {
        if (getSelectedTabId%2==0) {
            alert("感谢您的反馈！");
        }
        else {
            alert("Thanks for your feedback!");
        };
        str = "ERROR: " + word + ": " + str;
        $.get("/feedback/", { 'content': str, 'mode': 'FBS' });
    }
}
function addTag(i, m) {
    if (getSelectedTabId==0) {
        var word = $("#tabs-1 #li"+i).text();
        var description = $("#description").val();
        var elemA = $("#tabs-1 #li"+i);
        var elemD = $("#tabs-1 #li"+i+" span");
    }
    else if (getSelectedTabId==1) {
        var word = $("#tabs-2 #li"+i).text();
        var description = $("#description_EE").val();
        var elemA = $("#tabs-2 #li"+i);
        var elemD = $("#tabs-2 #li"+i+" span");
    }
    else if (getSelectedTabId==2) {
        var word = $("#tabs-3 #li"+i).text();
        var description = $("#description_CE").val();
        var elemA = $("#tabs-3 #li"+i);
        var elemD = $("#tabs-3 #li"+i+" span");
    }
    else if (getSelectedTabId==3) {
        var word = $("#tabs-4 #li"+i).text();
        var description = $("#description_EC").val();
        var elemA = $("#tabs-4 #li"+i);
        var elemD = $("#tabs-4 #li"+i+" span");
    }
    else {
        return null;
    };
    if (m==2) {elemA.append("<span style=\"color: red\" class=\"glyphicon glyphicon-thumbs-up\"></span>")}
    else if (m==1) {elemD.remove()}
    else if (m==0) {elemA.append("<span class=\"glyphicon glyphicon-thumbs-down\"></span>")};
    str = word + "|" + m;
    $.get("/feedback/", { 'content': str+"|||"+description, 'mode': 'FBW' });
}
function clearAlert() {
    var selID = getSelectedTabId + 1;
    var elem = $("#tabs-" + selID +" .alert");
    elem.remove();
    //elem.slideUp("fast");
    $("div.popover").hide(); //清理卡死的popover弹框
}

function clearFilter() {
    var selID = getSelectedTabId + 1;
    clearAlert();
    if (selID==1) {
        try {
            $("#filter_CN div").find("*").removeAttr("disabled");
            if ($("#description").val()=="") {
                $('#result').html("");
            }
            else {
                if ($("#description").val()==description_backup) {
                    showTable(retData_backup, $('#result'));
                }
                else {
                    modelProcecss();
                };
            };
        }
        catch(err) {
            $('#result').html("");
        };                            
        $("#filter_CN div").find("*").val(this.defaultValue).css("background-color", "");
        $("#filter_CN div.visible-xs").find("#POS_select_CC")[0].selectedIndex = 0;
        $("#filter_CN div.visible-lg").find("#POS_select_CC")[0].selectedIndex = 0;
        $("#filter_CN div.visible-xs").find("#main_select")[0].selectedIndex = 0;
        $("#filter_CN div.visible-lg").find("#main_select")[0].selectedIndex = 0;
        $("#filter_CN div.visible-xs").find("#rhyme_select_CC")[0].selectedIndex = 0;
        $("#filter_CN div.visible-lg").find("#rhyme_select_CC")[0].selectedIndex = 0;
    }
    else if (selID==2) {
        try {
            $("#filter_EE div").find("*").removeAttr("disabled");
            if ($("#description_EE").val()=="") {
                $('#result_EE').html("");
            }
            else {
                if ($("#description_EE").val()==description_backup_EE) {
                    showTable(retData_backup_EE, $('#result_EE'));
                }
                else {
                    modelProcecss_EE();
                };
            };
        }
        catch(err) {
            $('#result_EE').html("");
        };                            
        $("#filter_EE div").find("*").val(this.defaultValue).css("background-color", "");
        $("#filter_EE div.visible-xs").find("#POS_select_EE")[0].selectedIndex = 0;
        $("#filter_EE div.visible-lg").find("#POS_select_EE")[0].selectedIndex = 0;
        $("#filter_EE div.visible-xs").find("#main_select_EE")[0].selectedIndex = 0;
        $("#filter_EE div.visible-lg").find("#main_select_EE")[0].selectedIndex = 0;
    }
    else if (selID==3) {
        try {
            $("#filter_CE div").find("*").removeAttr("disabled");
            if ($("#description_CE").val()=="") {
                $('#result_CE').html("");
            }
            else {
                if ($("#description_CE").val()==description_backup_CE) {
                    showTable(retData_backup_CE, $('#result_CE'));
                }
                else {
                    modelProcecss_CE();
                };
            };
        }
        catch(err) {
            $('#result_CE').html("");
        };                            
        $("#filter_CE div").find("*").val(this.defaultValue).css("background-color", "");
        $("#filter_CE div.visible-xs").find("#POS_select_CE")[0].selectedIndex = 0;
        $("#filter_CE div.visible-lg").find("#POS_select_CE")[0].selectedIndex = 0;
        $("#filter_CE div.visible-xs").find("#main_select_CE")[0].selectedIndex = 0;
        $("#filter_CE div.visible-lg").find("#main_select_CE")[0].selectedIndex = 0;
    }
    else if (selID==4) {
        try {
            $("#filter_EC div").find("*").removeAttr("disabled");
            if ($("#description_EC").val()=="") {
                $('#result_EC').html("");
            }
            else {
                if ($("#description_EC").val()==description_backup_EC) {
                    showTable(retData_backup_EC, $('#result_EC'));
                }
                else {
                    modelProcecss_EC();
                };
            };
        }
        catch(err) {
            $('#result_EC').html("");
        };                            
        $("#filter_EC div").find("*").val(this.defaultValue).css("background-color", "");
        $("#filter_EC div.visible-xs").find("#POS_select_EC")[0].selectedIndex = 0;
        $("#filter_EC div.visible-lg").find("#POS_select_EC")[0].selectedIndex = 0;
        $("#filter_EC div.visible-xs").find("#main_select_EC")[0].selectedIndex = 0;
        $("#filter_EC div.visible-lg").find("#main_select_EC")[0].selectedIndex = 0;
        $("#filter_EC div.visible-xs").find("#rhyme_select_EC")[0].selectedIndex = 0;
        $("#filter_EC div.visible-lg").find("#rhyme_select_EC")[0].selectedIndex = 0;
    };

};


<!----------------------------全局--------------------------------->
var itemsPerCol = 20;

function htmlSuccess(str) {
    return '<div class="alert alert-success alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>' + str + '</div>';
};
function htmlInfo(str) {
    return '<div class="alert alert-info alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-info-sign"></span></button><strong>信息：</strong>' + str + '</div>';
};
function htmlWarning(str) {
    return '<div class="alert alert-warning alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-eye-open"></span></button><strong>警告！</strong>' + str + '</div>';
};
function htmlDanger(str) {
    return '<div class="alert alert-danger alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-warning-sign"></button><strong>错误！</strong>' + str + '</div>';
};
function htmlInfo_E(str) {
    return '<div class="alert alert-info alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-info-sign"></span></button><strong>Info: </strong>' + str + '</div>';
};
function htmlWarning_E(str) {
    return '<div class="alert alert-warning alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-eye-open"></span></button><strong>Caution! </strong>' + str + '</div>';
};
function htmlDanger_E(str) {
    return '<div class="alert alert-danger alert-dismissable"><button type="button" class="close" data-dismiss="alert" aria-hidden="true"><span class="glyphicon glyphicon-warning-sign"></button><strong>Error! </strong>' + str + '</div>';
};

function getContent(wdData, defi, i) {
    var reg = /[āáǎàōóǒòêēéěèīíǐìūúǔùǖǘǚǜü]/g; //[āáǎàōóǒòêēéěèīíǐìūúǔùǖǘǚǜüńňǹĀÅÀö∥ɡa-zA-Z•ɑ’]
    if (defi.replace(/br/g,'').replace(/strong/g,'').search(reg)>-1) {
        var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + defi + '<HR/><label title=\'在openhownet中查看该词的义原。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>查看义原</button></label>' + '&nbsp;<label title=\'查看百度汉语中的释义。\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://hanyu.baidu.com/s?wd=' + wdData['w'] + '\')&quot;>百度汉语</button></label>' + '&nbsp;<label title=\'如果您发现定义、拼音等存在错误时点此反馈。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>上报错误</button></label>';
    }
    else {
        var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + wdData['p'] + '<br>' + defi + '<HR/><label title=\'在openhownet中查看该词的义原。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>查看义原</button></label>' + '&nbsp;<label title=\'查看百度汉语中的释义。\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://hanyu.baidu.com/s?wd=' + wdData['w'] + '\')&quot;>百度汉语</button></label>' + '&nbsp;<label title=\'如果您发现定义、拼音等存在错误时点此反馈。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>上报错误</button></label>';
    };
    return htmlCont;
};
function getTitle(i) {
    //var htmlTitle = '<div class=&quot;btn-group&quot; data-toggle=&quot;buttons&quot;><label title=\'非常匹配\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 2' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> 😃</span></label><label title=\'基本相关\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 1' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;>😐</span></label><label title=\'完全无关\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 0' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> 🙁</span></label></div>';
    var htmlTitle = '<div class=&quot;btn-group&quot; data-toggle=&quot;buttons&quot;><label title=\'非常匹配\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 2' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> <span class=&quot;glyphicon glyphicon-thumbs-up&quot;></span></label><label title=\'完全无关\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 0' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> <span class=&quot;glyphicon glyphicon-thumbs-down&quot;></span></label></div>';
    return htmlTitle;
};
function getContent_E(wdData, defi, i) {
    var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + defi + '<HR/><label title=\'View sememes in OpenHownet.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>sememes</button></label>' + '&nbsp;<label title=\'Look up the word in Wiktionary.\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://en.wiktionary.org/wiki/' + wdData['w'] + '\')&quot;>Wiki</button></label>' + '&nbsp;<label title=\'If there are any mistakes, you can tell us.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>Report errors</button></label>';
    //var htmlCont = '<h4><strong>' + wdData['word'] + '</strong></h4>' + '1. <strong>adj. </strong>' + wdData['definition'] + '<br><HR/><label title=\'View sememes in OpenHownet.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['word'] + '\')&quot;>sememes</button></label>' + '&nbsp;<label title=\'Look up the word in Wiktionary.\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://en.wiktionary.org/wiki/' + wd_data['word'] + '\')&quot;>Wikipedia</button></label>' + '&nbsp;<label title=\'If there are any mistakes, you can tell us.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>Report errors</button></label>';
    return htmlCont;
};
function getTitle_E(i) {
    //var htmlTitle = '<div class=&quot;btn-group&quot; data-toggle=&quot;buttons&quot;><label title=\'Matched well\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 2' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> 😃</span></label><label title=\'So-so\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 1' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;>😐</span></label><label title=\'Not matched\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 0' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> 🙁</span></label></div>';
    var htmlTitle = '<div class=&quot;btn-group&quot; data-toggle=&quot;buttons&quot;><label title=\'Matched well\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 2' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> <span class=&quot;glyphicon glyphicon-thumbs-up&quot;></span></label><label title=\'Not matched\' class=&quot;btn btn-primary&quot; onclick=&quot;addTag(' + i + ', 0' + ')&quot; style=&quot;background-color: #eee; color: #333;&quot;><input type=&quot;radio&quot;> <span class=&quot;glyphicon glyphicon-thumbs-down&quot;></span></label></div>';
    return htmlTitle;
};
function getContent_CE(wdData, defi, i) {
    var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + defi + '<HR/><label title=\'在openhownet中查看该词的义原。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>查看义原</button></label>' + '&nbsp;<label title=\'查看维基词典中的释义。\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://en.wiktionary.org/wiki/' + wdData['w'] + '\')&quot;>维基词典</button></label>' + '&nbsp;<label title=\'如果您发现错误时请点此反馈。\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>上报错误</button></label>';
    return htmlCont;
};
function getContent_EC(wdData, defi, i) {
    var reg = /[āáǎàōóǒòêēéěèīíǐìūúǔùǖǘǚǜüńňǹĀÅÀö∥ɡa-zA-Z•ɑ’]/g;
    
    if (defi.replace(/br/g,'').replace(/strong/g,'').search(reg)>-1) {
        var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + defi + '<HR/><label title=\'View sememes in OpenHownet.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>sememes</button></label>' + '&nbsp;<label title=\'Look up the word in Baidu.\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://hanyu.baidu.com/s?wd=' + wdData['w'] + '\')&quot;>Baidu</button></label>' + '&nbsp;<label title=\'If there are any mistakes, you can tell us.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>Report errors</button></label>';
    }
    else {
        var htmlCont = '<h4><strong>' + wdData['w'] + '</strong></h4>' + wdData['p'] + '<br>' + defi + '<HR/><label title=\'View sememes in OpenHownet.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;window.open(\'https://openhownet.thunlp.org/search_list.html?keyword=' + wdData['w'] + '\')&quot;>sememes</button></label>' + '&nbsp;<label title=\'Look up the word in Baidu.\'><button class=&quot;btn btn-default  btn-sm&quot; onclick=&quot;window.open(\'https://hanyu.baidu.com/s?wd=' + wdData['w'] + '\')&quot;>Baidu</button></label>' + '&nbsp;<label title=\'If there are any mistakes, you can tell us.\'><button class=&quot;btn btn-default btn-sm&quot; onclick=&quot;diagError(' + i + ')&quot;>Report errors</button></label>';
    };
    return htmlCont;
};
function showTable(dictList, res_elem) {
    //$('div.popover').children().hide();
    var words = '';
    for (var d in dictList) {
        words = words + ' ' + dictList[d].w;
    };
    var desti = "/GetEnDefis/";
    if ('p' in dictList[0]) {
        desti = "/GetChDefis/";
    };
    
    $.post(desti, {'w': words}, function (ret) {
        var defis = ret.slice(0);
        var block_start = '<div class="col-xs-6 col-sm-4 col-md-3 col-lg-2">';
        var block_end = '</ol></div>';
        var html = '<div class="container"><div class="row" >';
        var i = 0;
        var num = dictList.length>100 ? 100 : dictList.length;
        for (; i<num; ){
            wd_data = dictList[i];
            if (i%itemsPerCol==0) {
                html += block_start;
                html = html + '<ol start="' + ((parseInt(i/itemsPerCol))*itemsPerCol).toString() + '" style="color:grey">';
            }
            if (getSelectedTabId==0) {
                if ($(window).width()<751 || window.innerWidth<768) {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop0" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                }
                else {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop0" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                };
            }
            else if (getSelectedTabId==1) {
                if ($(window).width()<751 || window.innerWidth<768) {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop1" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_E(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                }
                else {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop1" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_E(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                };
            }
            else if (getSelectedTabId==2) {
                if ($(window).width()<751 || window.innerWidth<768) {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop2" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_CE(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                }
                else {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop2" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_CE(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                };
            }
            else {
                if ($(window).width()<751 || window.innerWidth<768) {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop3" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_EC(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                }
                else {
                    html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop3" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_EC(wd_data, defis[i], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
                };
            };
            i += 1;
            if (i%itemsPerCol==0) {
                html += block_end;
            }
        };
        clearAlert();
        res_elem.html(html);
        if (getSelectedTabId==0) {
            $('a.pop0').click();
        }
        else if (getSelectedTabId==1) {
            $('a.pop1').click();
        }
        else if (getSelectedTabId==2) {
            $('a.pop2').click();
        }
        else {
            $('a.pop3').click();
        }
        //$('a.pop').click(); // 这里是用于对新生成的html进行绑定，因为HTML是静态代码，页面生成时绑定了js和html的关系（执行了js代码一次），但是这个新生成的html不被当时的js代码绑定，所以这里再执行一次js代码。
        //$('div.popover').children().hide();
        $('div.popover').hide();
        if (getSelectedTabId%2==0) {
            res_elem.before('<div class="alert alert-success alert-dismissable" style="font-family:STZhongsong;font-size:15px;"><button type="button" class="close" data-dismiss="alert" aria-hidden="true">×</button><strong>使用建议：</strong><br>1、配合筛选器使用，效果更佳。<br>2、按相关性排序或聚类排列可以把更相关的词排在前面。<br>3、点击词语显示详情，在详情框顶部可对该词点“赞”或“踩”。<br><span class="glyphicon glyphicon-heart" style="color:red;"></span> 欢迎多点评，将有助于为大家做出更精准的推荐服务 : )</div>');
            $(".alert").on("click", function(){$(this).slideUp("fast");});
        }
        else {
            res_elem.before('<div class="alert alert-success alert-dismissable" style="font-family:STZhongsong;font-size:15px;"><button type="button" class="close" data-dismiss="alert" aria-hidden="true">×</button><strong>Suggestions: </strong><br>1. Better results with filtering.<br>2. More relevant words can be ranked in the first place by ranking by relevance or clustering.<br>3. Click on a word to see details. You can mark a word <i>good</i> or <i>bad</i> at the top of the details box.<br><span class="glyphicon glyphicon-heart" style="color:red;"></span> Welcome to mark more words, which is helpful for more accurate recommendation : )</div>');
            $(".alert").on("click", function(){$(this).slideUp("fast");});
        };
    });
};

function showTable_Cluster(dictList, res_elem) {
    var block_start = '<div class="col-xs-6 col-sm-4 col-md-3 col-lg-2">';
    var block_end = '</ul></div>';
    var html = '<div class="container"><div class="row" >';
    var i = 0;
    var num = dictList.length;
    var itemsPerCol_ = 10;
    var count = 0;
    var Class = 0;
    var addFlag = true;
    for (; i<num; ){
        wd_data = dictList[i];
        /*if (count==itemsPerCol_){
            if (addFlag) {
                html += block_end;
            };
            if (Class-1==wd_data['C']) {
                i += 1;
                addFlag = false;
                continue;
            }
            else {
                addFlag = true;
                count = 0;
            };
        };*/
        if (Class==wd_data['C']) {
            if (Class>0) {
                html += block_end;
                count = 0;
            };
            Class += 1;
            html += block_start;
            html = html + '<ul style="color:grey;">';
        }
        //if ('p' in wd_data) { // 中文里有 wd_data['pinyin']。
        if (getSelectedTabId==0) {
            $("#filter_CN div").find("input").attr("disabled", "disabled");
            $("#filter_CN div.visible-xs").find("#POS_select_CC").attr("disabled", "disabled");
            $("#filter_CN div.visible-lg").find("#POS_select_CC").attr("disabled", "disabled");
            $("#filter_CN div.visible-xs").find("#rhyme_select_CC").attr("disabled", "disabled");
            $("#filter_CN div.visible-lg").find("#rhyme_select_CC").attr("disabled", "disabled");
            if ($(window).width()<751 || window.innerWidth<768) {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop0" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            }
            else {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop0" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            }
        }
        else if (getSelectedTabId==1) {
            $("#filter_EE div").find("input").attr("disabled", "disabled");
            $("#filter_EE div.visible-xs").find("#POS_select_EE").attr("disabled", "disabled");
            $("#filter_EE div.visible-lg").find("#POS_select_EE").attr("disabled", "disabled");
            if ($(window).width()<751 || window.innerWidth<768) {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop1" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_E(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            }
            else {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop1" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_E(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            };
        }
        else if (getSelectedTabId==2) {
            $("#filter_CE div").find("input").attr("disabled", "disabled");
            $("#filter_CE div.visible-xs").find("#POS_select_CE").attr("disabled", "disabled");
            $("#filter_CE div.visible-lg").find("#POS_select_CE").attr("disabled", "disabled");
            if ($(window).width()<751 || window.innerWidth<768) {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop2" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_CE(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            }
            else {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle(i) + '" class="pop2" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_CE(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            };
        }
        else {
            $("#filter_EC div").find("input").attr("disabled", "disabled");
            $("#filter_EC div.visible-xs").find("#POS_select_EC").attr("disabled", "disabled");
            $("#filter_EC div.visible-lg").find("#POS_select_EC").attr("disabled", "disabled");
            $("#filter_EC div.visible-xs").find("#rhyme_select_EC").attr("disabled", "disabled");
            $("#filter_EC div.visible-lg").find("#rhyme_select_EC").attr("disabled", "disabled");
            if ($(window).width()<751 || window.innerWidth<768) {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop3" data-container="body" data-placement="auto bottom" data-toggle="popover" data-content="' + getContent_EC(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            }
            else {
                html = html + '<li id=\"li' + i + '\" style=\"background-color: #005aff' + wd_data['c'] + ';"><a title="' + getTitle_E(i) + '" class="pop3" data-container="body" data-placement="auto right" data-toggle="popover" data-content="' + getContent_EC(wd_data, wd_data['d'], i) + '" style="color:black"><strong style="cursor:pointer">' + wd_data['w'] + '</strong></a></li>';
            };
        };
        i += 1;
        count += 1;
    };
    clearAlert();
    res_elem.html(html);
    if (getSelectedTabId==0) {
        $('a.pop0').click();
    }
    else if (getSelectedTabId==2) {
        $('a.pop1').click();
    }
    else if (getSelectedTabId==2) {
        $('a.pop2').click();
    }
    else {
        $('a.pop3').click();
    }
    //$('a.pop').click(); // 这里是用于对新生成的html进行绑定，因为HTML是静态代码，页面生成时绑定了js和html的关系（执行了js代码一次），但是这个新生成的html不被当时的js代码绑定，所以这里再执行一次js代码。
    $('div.popover').hide();
};

<!----------------------------汉汉CC--------------------------------->
var retData_backup; //全局变量保存返回值原始数据。
var description_backup;

//filterRes();
function filterRes(dictList) {
    //console.log("filterRes");
    //var filter_POS = $("#filter1").val(); //document.getElementById("filter1").value
    if ($(window).width()<751 || window.innerWidth<768) {
        var POS_select_CC=$("#filter_CN div.visible-xs").find("#POS_select_CC");
        var filter2=$("#filter_CN div.visible-xs").find("#filter2");
        var filter3=$("#filter_CN div.visible-xs").find("#filter3");
        var filter4=$("#filter_CN div.visible-xs").find("#filter4");
        var filter5=$("#filter_CN div.visible-xs").find("#filter5");
        var main_select=$("#filter_CN div.visible-xs").find("#main_select");
        var rhyme_select_CC=$("#filter_CN div.visible-xs").find("#rhyme_select_CC");
    }
    else {
        var POS_select_CC=$("#filter_CN div.visible-lg").find("#POS_select_CC");
        var filter2=$("#filter_CN div.visible-lg").find("#filter2");
        var filter3=$("#filter_CN div.visible-lg").find("#filter3");
        var filter4=$("#filter_CN div.visible-lg").find("#filter4");
        var filter5=$("#filter_CN div.visible-lg").find("#filter5");
        var main_select=$("#filter_CN div.visible-lg").find("#main_select");
        var rhyme_select_CC=$("#filter_CN div.visible-lg").find("#rhyme_select_CC");
    };
    //var filter_POS = document.getElementById("POS_select_CC").options.selectedIndex;
    var filter_POS = POS_select_CC[0].selectedIndex;
    var filter_len = filter2.val();
    var filter_1stPY = filter3.val();
    var filter_strok = filter4.val();
    var filter_shape = filter5.val();
    var sort_rule = main_select[0].selectedIndex;
    var filter_rhyme = rhyme_select_CC[0].selectedIndex;
    if (filter_POS>0) {
        POS_select_CC.css("background-color", "#fffdef");
    }
    else {
        POS_select_CC.css("background-color", "");
    };
    if (filter_len!="") {
        filter2.css("background-color", "#fffdef");
    }
    else {
        filter2.css("background-color", "");
    };
    if (filter_1stPY!="") {
        filter3.css("background-color", "#fffdef");
    }
    else {
        filter3.css("background-color", "");
    };
    if (filter_strok!="") {
        filter4.css("background-color", "#fffdef");
    }
    else {
        filter4.css("background-color", "");
    };
    if (filter_shape!="") {
        filter5.css("background-color", "#fffdef");
    }
    else {
        filter5.css("background-color", "");
    };
    if (sort_rule>0) {
        main_select.css("background-color", "#fffdef");
    }
    else {
        main_select.css("background-color", "");
    };
    if (filter_rhyme>0) {
        rhyme_select_CC.css("background-color", "#fffdef");
    }
    else {
        rhyme_select_CC.css("background-color", "");
    };
    switch (filter_POS) {
        case 0:
            var dictList_filtered = dictList.slice(0);
            break;
        case 1:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("名")>-1});
            break;                                                                              
        case 2:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("动")>-1});
            break;                                                                              
        case 3:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("形")>-1});
            break;                                                                              
        case 4:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("副")>-1});
            break;                                                                              
        case 5:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("介")>-1});
            break;                                                                              
        case 6:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("数")>-1});
            break;                                                                              
        case 7:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("连")>-1});
            break;                                                                              
        case 8:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("助")>-1});
            break;                                                                              
        case 9:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("量")>-1});
            break;                                                                              
        case 10:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("叹")>-1});
            break;                                                                              
        case 11:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("代")>-1});
            break;                                                                              
        case 12:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("拟声")>-1});
            break;
        case 13:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("无")>-1});
            break;
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_rhyme>0) {
        var dictList_filtered = dictList_filtered.filter(function (value) {return value.r.indexOf(filter_rhyme)>-1});
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_len != "") {
        if (filter_len>0 && filter_len<=8) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l == filter_len;
            };
        }
        else if (filter_len.indexOf('>')>-1 && filter_len.slice(filter_len.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l > filter_len.slice(filter_len.indexOf('>')+1);
            };
            filter2.val(">" + filter_len.slice(filter_len.indexOf('>')+1));
        }
        else if (filter_len.indexOf('<')>-1 && filter_len.slice(filter_len.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l < filter_len.slice(filter_len.indexOf('<')+1);
            };
            filter2.val("<" + filter_len.slice(filter_len.indexOf('<')+1));
        }
        else {
            //警告框
            $("#filter_CN").after(htmlWarning("字数筛选条件 “"+filter_len+"” 超出范围或无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter2.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_1stPY != "") {
        /*if (filter_1stPY>='A' && filter_1stPY<='z') {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w[0] == filter_1stPY[0];
            };
            document.getElementById("filter3").value = filter_1stPY[0].toLowerCase();
        }*/
        filter_1stPY = filter_1stPY.toLowerCase();
        var reg = /[a-z]/g;
        if (filter_1stPY.replace(reg, "")=="") { //证明只有英文字母
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
                var pyszm = value.s.split(" ");
                for (var i=0;i<filter_1stPY.length;i++) {
                    if (pyszm[i]!=filter_1stPY[i]) {return false;};
                };
                return true;
            };
            filter3.val(filter_1stPY);
        }
        else {
            //警告框
            $("#filter_CN").after(htmlWarning("拼音首字母筛选条件 “"+filter_1stPY+"” 无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter3.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_strok != "") {
        if (filter_strok>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b == filter_strok;
            };
        }
        else if (filter_strok.indexOf('>')>-1 && filter_strok.slice(filter_strok.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b > filter_strok.slice(filter_strok.indexOf('>')+1);
            };
            filter4.val(">" + filter_strok.slice(filter_strok.indexOf('>')+1));
        }
        else if (filter_strok.indexOf('<')>-1 && filter_strok.slice(filter_strok.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b < filter_strok.slice(filter_strok.indexOf('<')+1);
            };
            filter4.val("<" + filter_strok.slice(filter_strok.indexOf('<')+1));
        }
        else {
            //警告框
            $("#filter_CN").after(htmlWarning("笔画筛选条件 “"+filter_strok+"” 无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter4.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    //*为匹配0到多字；？匹配1字；+为且；[...]匹配集合内任一字；[^...]不匹配集合内任何字
    if (filter_shape != "") {
        var reg = /[\u4e00-\u9fa5]/g;
        var ruleStr = "或********或????????或？？？？？？？？或++++++++或[^]或[]"; //多次匹配模式（第一个“或”字占位符必须加，因为如果搜索目标是空的则搜索结果是位置0）
        //var ruleStr = "或*或?或？或+或[^]或[]"; //单次匹配模式
        var ruleInd = ruleStr.indexOf(filter_shape.replace(reg, ""));
        var tmp = filter_shape.match(reg);
        try {
            var hanziStr = tmp.join("");
        }
        catch(err) {
            var hanziStr = "";
        };
        if (ruleInd>-1) {
            if (ruleStr[ruleInd]=='*') {
                var hanziArr = filter_shape.split('*');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    var tmp = [];
                    for (var i=0;i<this.length;i++) { // 山*水* --> ["山","水",""]，有一个空，因为*在边上的原因。
                        if (this[i].length>0) {
                            tmp.push(this[i]);
                        };
                    };
                    if (tmp.length==0) { return true;}; //没有汉字，则都算符合。
                    if (this[0]!="") { // 开头不是*而是字时，必须匹配第一个字/词
                        if (value.w[0]!=this[0]) {return false;};
                    };
                    if (this[this.length-1]!="") { // 结尾不是*而是字时，必须匹配最后一个字/词
                        if (value.w[value.w.length-1]!=this[this.length-1]) {return false;};
                    };
                    if (tmp.length==1) { //一个字或词，找到就符合。
                        if (value.w.indexOf(tmp[0])>-1) {
                            return true;
                        }
                        else {
                            return false;
                        };
                    }
                    else {
                        var ind = value.w.indexOf(tmp[0]);
                        if (ind<0) {return false;};
                        for (var i=1;i<tmp.length;i++) { //多个字或词，从上一次找到的点往后找，以保证按顺序。
                            if (value.w.indexOf(tmp[i], ind+1)<0) {
                                return false;
                            }
                            else {
                                ind = value.w.indexOf(tmp[i]);
                            };
                        };
                        return true;
                    };
                }, hanziArr);
            }
            else if (ruleStr[ruleInd]=='?' || ruleStr[ruleInd]=='？') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    if (filter_shape.length!=value.w.length) {return false};
                    for (var i=0;i<filter_shape.length;i++) {
                        if (filter_shape[i]==ruleStr[ruleInd]) {continue;}
                        else {
                            if (filter_shape[i]!=value.w[i]) {return false;};
                        };
                    };
                    return true;
                });
            }
            else if (ruleStr[ruleInd]=='+') {
                var hanziArr = filter_shape.split('+');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])<0) {return false;};
                    };
                    return true;
                }, hanziArr);
            }
            else if (ruleStr[ruleInd]=='[' && ruleStr[ruleInd+1]=='^') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])>-1) {return false;};
                    };
                    return true;
                }, hanziStr);
            }
            else if (ruleStr[ruleInd]=='[') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])>-1) {return true;};
                    };
                    return false;
                }, hanziStr);
            }
            else {
                //警告框
                $("#filter_CN").after(htmlWarning("词形筛选条件 “"+filter_shape+"” 无法识别。"));
                $(".alert").on("click", function(){$(this).slideUp("fast");});
                filter5.val(this.defaultValue);
                return false;
            };
        }
        else {
            //警告框
            $("#filter_CN").after(htmlWarning("词形筛选条件 “"+filter_shape+"” 无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter5.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CN").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    dictList_filtered = dictList_filtered.slice(0,100);
    switch (sort_rule) {
        case 1:
            dictList_filtered.sort(function(a, b){
                if (a.s[0] > b.s[0]) {
                    return 1;
                }
                else if (a.s[0] < b.s[0]) {
                    return -1;
                }
                else {
                    return 0;
                }
            });
            break;
        case 2:
            dictList_filtered.sort(function(a, b){
                if (a.s[0] > b.s[0]) {
                    return -1;
                }
                else if (a.s[0] < b.s[0]) {
                    return 1;
                }
                else {
                    return 0;
                }
            });
            break;
        case 3:
            dictList_filtered.sort(function(a, b){return a.b - b.b});
            break;
        case 4:
            dictList_filtered.sort(function(a, b){return b.b - a.b});
            break;
        case 5:
            dictList_filtered.sort(function(a, b){return a.B - b.B});
            break;
        case 6:
            dictList_filtered.sort(function(a, b){return b.B - a.B});
            break;
    };
    showTable(dictList_filtered, $('#result'));
};


function modelProcecss() {
    clearAlert();
    var description = $("#description").val();
    if (description.length==0) {
        $("#filter_CN").after(htmlDanger("输入描述不能为空。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    var reg = /[\u4e00-\u9fa5]/g;
    if (description.search(reg)<0) {
        $("#filter_CN").after(htmlDanger("输入字符无法识别。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    
    //聚类功能
    if ($(window).width()<751 || window.innerWidth<768) {
        var main_select=$("#filter_CN div.visible-xs").find("#main_select");
    }
    else {
        var main_select=$("#filter_CN div.visible-lg").find("#main_select"); 
    };
    var sort_rule = main_select[0].selectedIndex;
    if (sort_rule==7) {
        $.get("/ChineseRDCluster/", { 'description': description, 'mode': 'CC' }, function (ret) {
            showTable_Cluster(ret, $('#result'));
        });
        return true;
    }
    $("#filter_CN div").find("*").removeAttr("disabled");
    //console.log('modelProcecss');
    if ($("#description").val()==description_backup) {
        filterRes(retData_backup);
    }
    else {
        $.get("/ChineseRD/", { 'description': description, 'mode': 'CC' }, function (ret) {
            try {
                retData_backup = ret.slice(0);
                description_backup = description.slice(0);
                filterRes(retData_backup);
                $("#filter_CN").show();
            }
            catch(err) {
                $('#result').html("");
                switch (ret['error']){
                    case 0: //错误框
                        $("#filter_CN").after(htmlDanger("输入描述不能为空。"));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    case 1: //错误框
                        $("#filter_CN").after(htmlDanger("输入字符无法识别。"));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    default: //报出明确的错误类型。
                        $("#filter_CN").after(htmlDanger(err.name));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                }
            }
        });
    }
};
function onkeySearch() {
    $('#result').html("");
    clearAlert();
    modelProcecss();
};
$(document).ready(function () {
    $("#description").keypress(function(e) {
        if(e.keyCode == 13)
            {
                $('#result').html("");
                clearAlert();
                modelProcecss();
            }
    });
});
<!----------------------------英英EE--------------------------------->        
var retData_backup_EE; //全局变量保存返回值原始数据。
var description_backup_EE;

function filterRes_EE(dictList) {
    if ($(window).width()<751 || window.innerWidth<768) {
        var POS_select_EE=$("#filter_EE div.visible-xs").find("#POS_select_EE");
        var filter1=$("#filter_EE div.visible-xs").find("#filter1_EE");
        var filter2=$("#filter_EE div.visible-xs").find("#filter2_EE");
        var filter3=$("#filter_EE div.visible-xs").find("#filter3_EE");
        var main_select=$("#filter_EE div.visible-xs").find("#main_select_EE");
    }
    else {
        var POS_select_EE=$("#filter_EE div.visible-lg").find("#POS_select_EE");
        var filter1=$("#filter_EE div.visible-lg").find("#filter1_EE");
        var filter2=$("#filter_EE div.visible-lg").find("#filter2_EE");
        var filter3=$("#filter_EE div.visible-lg").find("#filter3_EE");
        var main_select=$("#filter_EE div.visible-lg").find("#main_select_EE");
    };
    var filter_POS = POS_select_EE[0].selectedIndex;
    var filter_len = filter1.val();
    var filter_initial = filter2.val();
    var filter_shape = filter3.val();
    var sort_rule = main_select[0].selectedIndex;
    if (filter_POS>0) {
        POS_select_EE.css("background-color", "#fffdef");
    }
    else {
        POS_select_EE.css("background-color", "");
    };
    if (filter_len!="") {
        filter1.css("background-color", "#fffdef");
    }
    else {
        filter1.css("background-color", "");
    };
    if (filter_initial!="") {
        filter2.css("background-color", "#fffdef");
    }
    else {
        filter2.css("background-color", "");
    };
    if (filter_shape!="") {
        filter3.css("background-color", "#fffdef");
    }
    else {
        filter3.css("background-color", "");
    };
    if (sort_rule>0) {
        main_select.css("background-color", "#fffdef");
    }
    else {
        main_select.css("background-color", "");
    };
    switch (filter_POS) {
        case 1:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("n")>-1});
            break;
        case 2:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("v")>-1});
            break;
        case 3:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("adj")>-1});
            break;
        case 4:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("adv")>-1});
            break;
        case 5:
            var dictList_filtered = dictList.filter(function (value) {return value.P.length==0});
            break;
        case 0:
            var dictList_filtered = dictList.slice(0);
            break;
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EE").after(htmlInfo_E("No screening results, please modify the POS screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_len != "") {
        if (filter_len>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length == filter_len;
            };
        }
        else if (filter_len.indexOf('>')>-1 && filter_len.slice(filter_len.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length > filter_len.slice(filter_len.indexOf('>')+1);
            };
            filter1.val(">" + filter_len.slice(filter_len.indexOf('>')+1));
        }
        else if (filter_len.indexOf('<')>-1 && filter_len.slice(filter_len.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length < filter_len.slice(filter_len.indexOf('<')+1);
            };
            filter1.val("<" + filter_len.slice(filter_len.indexOf('<')+1));
        }
        else {
            //警告框
            $("#filter_EE").after(htmlWarning_E("Word length screening condition '"+filter_len+"' is out of range or unrecognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter1.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EE").after(htmlInfo_E("No screening results, please modify the word length screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_initial != "") {
        var reg = /[a-zA-Z]/g;
        if (filter_initial.replace(reg, "")=="") { //证明只有英文字母
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
                return value.w[0] == filter_initial[0].toLowerCase();
            };
            filter2.val(filter_initial[0]);
        }
        else {
            //警告框
            $("#filter_EE").after(htmlWarning_E("Word initial screening condition '"+filter_initial+"' is not recognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter2.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EE").after(htmlInfo_E("No screening results, please modify the initial screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    //*为匹配0到多字；？匹配1字
    if (filter_shape != "") {
        var reg = /[a-zA-Z]/g;
        var ruleStr = "或****************或????????????????或？？？？？？？？？？？？？？？？"; //多次匹配模式（第一个“或”字占位符必须加，因为如果搜索目标是空的则搜索结果是位置0）
        var ruleInd = ruleStr.indexOf(filter_shape.replace(reg, ""));
        if (ruleInd>-1) {
            if (ruleStr[ruleInd]=='*') {
                var charArr = filter_shape.split('*');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    var tmp = [];
                    for (var i=0;i<this.length;i++) { // dic*on* --> ["dic","on",""]，有一个空，因为*在边上的原因。
                        if (this[i].length>0) {
                            tmp.push(this[i]);
                        };
                    };
                    if (tmp.length==0) { return true;}; //没有字母，则都算符合。
                    if (this[0]!="") { // 开头不是*而是字时，必须匹配第一个字母片段。########0814修改BUG：value.w[0]!=this[0]错在字母和字母片段进行对比。而是匹配第一个字母片段的首字母。
                        if (value.w[0]!=this[0][0]) {return false;};
                    };
                    if (this[this.length-1]!="") { // 结尾不是*而是字时，必须匹配最后一个字母片段。########0814修改BUG：value.w[0]!=this[0]错在字母和字母片段进行对比。而是匹配末字母片段的末字母。
                        if (value.w[value.w.length-1]!=this[this.length-1][this[this.length-1].length-1]) {return false;};
                    };
                    if (tmp.length==1) { //一个字母片段，找到就符合。
                        if (value.w.indexOf(tmp[0])>-1) {
                            return true;
                        }
                        else {
                            return false;
                        };
                    }
                    else {
                        var ind = value.w.indexOf(tmp[0]);
                        if (ind<0) {return false;};
                        for (var i=1;i<tmp.length;i++) { //多个字母片段，从上一次找到的点往后找，以保证按顺序。
                            if (value.w.indexOf(tmp[i], ind+1)<0) {
                                return false;
                            }
                            else {
                                ind = value.w.indexOf(tmp[i]);
                            };
                        };
                        return true;
                    };
                }, charArr);
            }
            else if (ruleStr[ruleInd]=='?' || ruleStr[ruleInd]=='？') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    if (filter_shape.length!=value.w.length) {return false};
                    for (var i=0;i<filter_shape.length;i++) {
                        if (filter_shape[i]==ruleStr[ruleInd]) {continue;}
                        else {
                            if (filter_shape[i]!=value.w[i]) {return false;};
                        };
                    };
                    return true;
                });
            }
            else {
                //警告框
                $("#filter_EE").after(htmlWarning_E("Wildcard patterns screening condition '"+filter_shape+"' is not recognizable."));
                $(".alert").on("click", function(){$(this).slideUp("fast");});
                filter3.val(this.defaultValue);
                return false;
            };
        }
        else {
            //警告框
            $("#filter_EE").after(htmlWarning_E("Wildcard patterns screening condition '"+filter_shape+"' is not recognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter3.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EE").after(htmlInfo_E("No screening results, please modify the Wildcard patterns screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    dictList_filtered = dictList_filtered.slice(0,100);
    switch (sort_rule) {
        case 1:
            dictList_filtered.sort(function(a, b){
                if (a.w[0] > b.w[0]) {
                    return 1;
                }
                else if (a.w[0] < b.w[0]) {
                    return -1;
                }
                else {
                    if (a.w[1] > b.w[1]) {
                        return 1;
                    }
                    else if (a.w[1] < b.w[1]) {
                        return -1;
                    }
                    else {
                        return 0;
                    }
                }
            });
            break;
        case 2:
            dictList_filtered.sort(function(a, b){
                if (a.w[0] > b.w[0]) {
                    return -1;
                }
                else if (a.w[0] < b.w[0]) {
                    return 1;
                }
                else {
                    if (a.w[1] > b.w[1]) {
                        return -1;
                    }
                    else if (a.w[1] < b.w[1]) {
                        return 1;
                    }
                    else {
                        return 0;
                    }
                }
            });
            break;
        case 3:
            dictList_filtered.sort(function(a, b){return a.w.length - b.w.length});
            break;
        case 4:
            dictList_filtered.sort(function(a, b){return b.w.length - a.w.length});
            break;
    };
    showTable(dictList_filtered, $('#result_EE'));
};


function modelProcecss_EE() {
    clearAlert();
    var description = $("#description_EE").val();
    if (description.length==0) {
        $("#filter_EE").after(htmlDanger_E("The input description cannot be empty."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    var reg = /[a-zA-Z]/;
    if (description.search(reg)<0) {
        $("#filter_EE").after(htmlDanger_E("The input characters are unrecognizable."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    
    // 聚类功能
    if ($(window).width()<751 || window.innerWidth<768) {
        var main_select=$("#filter_EE div.visible-xs").find("#main_select_EE");
    }
    else {
        var main_select=$("#filter_EE div.visible-lg").find("#main_select_EE"); 
    };
    var sort_rule = main_select[0].selectedIndex;
    if (sort_rule==5) {
        $.get("/EnglishRDCluster/", { 'description': description, 'mode': 'EE' }, function (ret) {
            showTable_Cluster(ret, $('#result_EE'));
        });
        return true;
    }
    $("#filter_EE div").find("*").removeAttr("disabled");
    if ($("#description_EE").val()==description_backup_EE) {
        filterRes_EE(retData_backup_EE);
    }
    else {
        $.get("/EnglishRD/", { 'description': description, 'mode': 'EE' }, function (ret) {
            try {
                retData_backup_EE = ret.slice(0);
                description_backup_EE = description.slice(0);
                //console.log(ret);
                filterRes_EE(retData_backup_EE);
                $("#filter_EE").show();
            }
            catch(err) {
                $('#result_EE').html("");
                switch (ret['error']){
                    case 0: //错误框
                        $("#filter_EE").after(htmlDanger_E("The input description cannot be empty."));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    case 1: //错误框
                        $("#filter_EE").after(htmlDanger_E("The input characters are unrecognizable."));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    default: //报出明确的错误类型。
                        $("#filter_EE").after(htmlDanger_E(err.name));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                }
            }
        });
    }
};
function onkeySearch_EE() {
    $('#result_EE').html("");
    clearAlert();
    modelProcecss_EE();
};
$(document).ready(function () {
    $("#description_EE").keypress(function(e) {
        if(e.keyCode == 13)
            {
                $('#result_EE').html("");
                clearAlert();
                modelProcecss_EE();
            }
    });
});

<!----------------------------汉英CE--------------------------------->
var retData_backup_CE; //全局变量保存返回值原始数据。
var description_backup_CE;

function filterRes_CE(dictList) {
    if ($(window).width()<751 || window.innerWidth<768) {
        var POS_select_CE=$("#filter_CE div.visible-xs").find("#POS_select_CE");
        var filter1=$("#filter_CE div.visible-xs").find("#filter1_CE");
        var filter2=$("#filter_CE div.visible-xs").find("#filter2_CE");
        var filter3=$("#filter_CE div.visible-xs").find("#filter3_CE");
        var main_select=$("#filter_CE div.visible-xs").find("#main_select_CE");
    }
    else {
        var POS_select_CE=$("#filter_CE div.visible-lg").find("#POS_select_CE");
        var filter1=$("#filter_CE div.visible-lg").find("#filter1_CE");
        var filter2=$("#filter_CE div.visible-lg").find("#filter2_CE");
        var filter3=$("#filter_CE div.visible-lg").find("#filter3_CE");
        var main_select=$("#filter_CE div.visible-lg").find("#main_select_CE");
    };
    var filter_POS = POS_select_CE[0].selectedIndex;
    var filter_len = filter1.val();
    var filter_initial = filter2.val();
    var filter_shape = filter3.val();
    var sort_rule = main_select[0].selectedIndex;
    if (filter_POS>0) {
        POS_select_CE.css("background-color", "#fffdef");
    }
    else {
        POS_select_CE.css("background-color", "");
    };
    if (filter_len!="") {
        filter1.css("background-color", "#fffdef");
    }
    else {
        filter1.css("background-color", "");
    };
    if (filter_initial!="") {
        filter2.css("background-color", "#fffdef");
    }
    else {
        filter2.css("background-color", "");
    };
    if (filter_shape!="") {
        filter3.css("background-color", "#fffdef");
    }
    else {
        filter3.css("background-color", "");
    };
    if (sort_rule>0) {
        main_select.css("background-color", "#fffdef");
    }
    else {
        main_select.css("background-color", "");
    };
    switch (filter_POS) {
        case 1:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("n")>-1});
            break;
        case 2:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("v")>-1});
            break;
        case 3:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("adj")>-1});
            break;
        case 4:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("adv")>-1});
            break;
        case 5:
            var dictList_filtered = dictList.filter(function (value) {return value.P.length==0});
            break;
        case 0:
            var dictList_filtered = dictList.slice(0);
            break;
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CE").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_len != "") {
        if (filter_len>0 && filter_len<=30) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length == filter_len;
            };
        }
        else if (filter_len.indexOf('>')>-1 && filter_len.slice(filter_len.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length > filter_len.slice(filter_len.indexOf('>')+1);
            };
            filter1.val(">" + filter_len.slice(filter_len.indexOf('>')+1));
        }
        else if (filter_len.indexOf('<')>-1 && filter_len.slice(filter_len.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.w.length < filter_len.slice(filter_len.indexOf('<')+1);
            };
            filter1.val("<" + filter_len.slice(filter_len.indexOf('<')+1));
        }
        else {
            //警告框
            $("#filter_CE").after(htmlWarning("单词长度筛选条件 '"+filter_len+"' 超出范围或无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter1.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CE").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_initial != "") {
        var reg = /[a-zA-Z]/g;
        if (filter_initial.replace(reg, "")=="") { //证明只有英文字母
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
                return value.w[0] == filter_initial[0].toLowerCase();
            };
            filter2.val(filter_initial[0]);
        }
        else {
            //警告框
            $("#filter_CE").after(htmlWarning("单词首字母筛选条件 '"+filter_initial+"' 无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter2.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CE").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    //*为匹配0到多字；？匹配1字
    if (filter_shape != "") {
        var reg = /[a-zA-Z]/g;
        var ruleStr = "或****************或????????????????或？？？？？？？？？？？？？？？？"; //多次匹配模式（第一个“或”字占位符必须加，因为如果搜索目标是空的则搜索结果是位置0）
        var ruleInd = ruleStr.indexOf(filter_shape.replace(reg, ""));
        if (ruleInd>-1) {
            if (ruleStr[ruleInd]=='*') {
                var charArr = filter_shape.split('*');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    var tmp = [];
                    for (var i=0;i<this.length;i++) { // dic*on* --> ["dic","on",""]，有一个空，因为*在边上的原因。
                        if (this[i].length>0) {
                            tmp.push(this[i]);
                        };
                    };
                    if (tmp.length==0) { return true;}; //没有字母，则都算符合。
                    if (this[0]!="") { // 开头不是*而是字时，必须匹配第一个字母片段。########0814修改BUG：value.w[0]!=this[0]错在字母和字母片段进行对比。而是匹配第一个字母片段的首字母。
                        if (value.w[0]!=this[0][0]) {return false;};
                    };
                    if (this[this.length-1]!="") { // 结尾不是*而是字时，必须匹配最后一个字母片段。########0814修改BUG：value.w[0]!=this[0]错在字母和字母片段进行对比。而是匹配末字母片段的末字母。
                        if (value.w[value.w.length-1]!=this[this.length-1][this[this.length-1].length-1]) {return false;};
                    };
                    if (tmp.length==1) { //一个字母片段，找到就符合。
                        if (value.w.indexOf(tmp[0])>-1) {
                            return true;
                        }
                        else {
                            return false;
                        };
                    }
                    else {
                        var ind = value.w.indexOf(tmp[0]);
                        if (ind<0) {return false;};
                        for (var i=1;i<tmp.length;i++) { //多个字母片段，从上一次找到的点往后找，以保证按顺序。
                            if (value.w.indexOf(tmp[i], ind+1)<0) {
                                return false;
                            }
                            else {
                                ind = value.w.indexOf(tmp[i]);
                            };
                        };
                        return true;
                    };
                }, charArr);
            }
            else if (ruleStr[ruleInd]=='?' || ruleStr[ruleInd]=='？') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    if (filter_shape.length!=value.w.length) {return false};
                    for (var i=0;i<filter_shape.length;i++) {
                        if (filter_shape[i]==ruleStr[ruleInd]) {continue;}
                        else {
                            if (filter_shape[i]!=value.w[i]) {return false;};
                        };
                    };
                    return true;
                });
            }
            else {
                //警告框
                $("#filter_CE").after(htmlWarning("词形筛选条件 '"+filter_shape+"' 无法识别。"));
                $(".alert").on("click", function(){$(this).slideUp("fast");});
                filter3.val(this.defaultValue);
                return false;
            };
        }
        else {
            //警告框
            $("#filter_CE").after(htmlWarning("词形筛选条件 '"+filter_shape+"' 无法识别。"));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter3.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_CE").after(htmlInfo("无筛选结果，请修改筛选条件。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    dictList_filtered = dictList_filtered.slice(0,100);
    switch (sort_rule) {
        case 1:
            dictList_filtered.sort(function(a, b){
                if (a.w[0] > b.w[0]) {
                    return 1;
                }
                else if (a.w[0] < b.w[0]) {
                    return -1;
                }
                else {
                    if (a.w[1] > b.w[1]) {
                        return 1;
                    }
                    else if (a.w[1] < b.w[1]) {
                        return -1;
                    }
                    else {
                        return 0;
                    }
                }
            });
            break;
        case 2:
            dictList_filtered.sort(function(a, b){
                if (a.w[0] > b.w[0]) {
                    return -1;
                }
                else if (a.w[0] < b.w[0]) {
                    return 1;
                }
                else {
                    if (a.w[1] > b.w[1]) {
                        return -1;
                    }
                    else if (a.w[1] < b.w[1]) {
                        return 1;
                    }
                    else {
                        return 0;
                    }
                }
            });
            break;
        case 3:
            dictList_filtered.sort(function(a, b){return a.w.length - b.w.length});
            break;
        case 4:
            dictList_filtered.sort(function(a, b){return b.w.length - a.w.length});
            break;
    };
    showTable(dictList_filtered, $('#result_CE'));
};


function modelProcecss_CE() {
    clearAlert();
    var description = $("#description_CE").val();
    if (description.length==0) {
        $("#filter_CE").after(htmlDanger("输入描述不能为空。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    var reg = /[\u4e00-\u9fa5]/g;
    if (description.search(reg)<0) {
        $("#filter_CE").after(htmlDanger("输入字符无法识别。"));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    
    // 聚类功能
    if ($(window).width()<751 || window.innerWidth<768) {
        var main_select=$("#filter_CE div.visible-xs").find("#main_select_CE");
    }
    else {
        var main_select=$("#filter_CE div.visible-lg").find("#main_select_CE"); 
    };
    var sort_rule = main_select[0].selectedIndex;
    if (sort_rule==5) {
        $.get("/EnglishRDCluster/", { 'description': description, 'mode': 'CE' }, function (ret) {
            showTable_Cluster(ret, $('#result_CE'));
        });
        return true;
    }
    $("#filter_CE div").find("*").removeAttr("disabled");
    if ($("#description_CE").val()==description_backup_CE) {
        filterRes_CE(retData_backup_CE);
    }
    else {
        $.get("/EnglishRD/", { 'description': description, 'mode': 'CE' }, function (ret) {
            try {
                retData_backup_CE = ret.slice(0);
                description_backup_CE = description.slice(0);
                //console.log(ret);
                filterRes_CE(retData_backup_CE);
                $("#filter_CE").show();
            }
            catch(err) {
                $('#result_CE').html("");
                switch (ret['error']){
                    case 0: //错误框
                        $("#filter_CE").after(htmlDanger("输入描述不能为空。"));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    case 1: //错误框
                        $("#filter_CE").after(htmlDanger("输入字符无法识别。"));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    default: //报出明确的错误类型。
                        $("#filter_CE").after(htmlDanger(err.name));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                }
            }
        });
    }
};
function onkeySearch_CE() {
    $('#result_CE').html("");
    clearAlert();
    modelProcecss_CE();
};
$(document).ready(function () {
    $("#description_CE").keypress(function(e) {
        if(e.keyCode == 13)
            {
                $('#result_CE').html("");
                clearAlert();
                modelProcecss_CE();
            }
    });
});
<!----------------------------英汉EC--------------------------------->
var retData_backup_EC; //全局变量保存返回值原始数据。
var description_backup_EC;

//filterRes_EC();
function filterRes_EC(dictList) {
    //console.log("filterRes_EC");
    //var filter_POS = $("#filter1_EC").val(); //document.getElementById("filter1").value
    if ($(window).width()<751 || window.innerWidth<768) {
        var POS_select_EC=$("#filter_EC div.visible-xs").find("#POS_select_EC");
        var filter2=$("#filter_EC div.visible-xs").find("#filter2_EC");
        var filter3=$("#filter_EC div.visible-xs").find("#filter3_EC");
        var filter4=$("#filter_EC div.visible-xs").find("#filter4_EC");
        var filter5=$("#filter_EC div.visible-xs").find("#filter5_EC");
        var main_select=$("#filter_EC div.visible-xs").find("#main_select_EC");
        var rhyme_select_EC=$("#filter_EC div.visible-xs").find("#rhyme_select_EC");
    }
    else {
        var POS_select_EC=$("#filter_EC div.visible-lg").find("#POS_select_EC");
        var filter2=$("#filter_EC div.visible-lg").find("#filter2_EC");
        var filter3=$("#filter_EC div.visible-lg").find("#filter3_EC");
        var filter4=$("#filter_EC div.visible-lg").find("#filter4_EC");
        var filter5=$("#filter_EC div.visible-lg").find("#filter5_EC");
        var main_select=$("#filter_EC div.visible-lg").find("#main_select_EC");
        var rhyme_select_EC=$("#filter_EC div.visible-lg").find("#rhyme_select_EC");
    };
    //var filter_POS = document.getElementById("POS_select_CC").options.selectedIndex;
    var filter_POS = POS_select_EC[0].selectedIndex;
    var filter_len = filter2.val();
    var filter_1stPY = filter3.val();
    var filter_strok = filter4.val();
    var filter_shape = filter5.val();
    var sort_rule = main_select[0].selectedIndex;
    var filter_rhyme = rhyme_select_EC[0].selectedIndex;
    if (filter_POS>0) {
        POS_select_EC.css("background-color", "#fffdef");
    }
    else {
        POS_select_EC.css("background-color", "");
    };
    if (filter_len!="") {
        filter2.css("background-color", "#fffdef");
    }
    else {
        filter2.css("background-color", "");
    };
    if (filter_1stPY!="") {
        filter3.css("background-color", "#fffdef");
    }
    else {
        filter3.css("background-color", "");
    };
    if (filter_strok!="") {
        filter4.css("background-color", "#fffdef");
    }
    else {
        filter4.css("background-color", "");
    };
    if (filter_shape!="") {
        filter5.css("background-color", "#fffdef");
    }
    else {
        filter5.css("background-color", "");
    };
    if (sort_rule>0) {
        main_select.css("background-color", "#fffdef");
    }
    else {
        main_select.css("background-color", "");
    };
    if (filter_rhyme>0) {
        rhyme_select_EC.css("background-color", "#fffdef");
    }
    else {
        rhyme_select_EC.css("background-color", "");
    };
    switch (filter_POS) {
        case 0:
            var dictList_filtered = dictList.slice(0);
            break;
        case 1:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("名")>-1});
            break;                                                                              
        case 2:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("动")>-1});
            break;                                                                              
        case 3:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("形")>-1});
            break;                                                                              
        case 4:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("副")>-1});
            break;                                                                              
        case 5:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("介")>-1});
            break;                                                                              
        case 6:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("数")>-1});
            break;                                                                              
        case 7:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("连")>-1});
            break;                                                                              
        case 8:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("助")>-1});
            break;                                                                              
        case 9:                                                                                 
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("量")>-1});
            break;                                                                              
        case 10:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("叹")>-1});
            break;                                                                              
        case 11:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("代")>-1});
            break;                                                                              
        case 12:                                                                                
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("拟声")>-1});
            break;
        case 13:
            var dictList_filtered = dictList.filter(function (value) {return value.P.indexOf("无")>-1});
            break;
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the POS screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_rhyme>0) {
        var dictList_filtered = dictList_filtered.filter(function (value) {return value.r.indexOf(filter_rhyme)>-1});
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the rhyme screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_len != "") {
        if (filter_len>0 && filter_len<=8) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l == filter_len;
            };
        }
        else if (filter_len.indexOf('>')>-1 && filter_len.slice(filter_len.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l > filter_len.slice(filter_len.indexOf('>')+1);
            };
            filter2.val(">" + filter_len.slice(filter_len.indexOf('>')+1));
        }
        else if (filter_len.indexOf('<')>-1 && filter_len.slice(filter_len.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.l < filter_len.slice(filter_len.indexOf('<')+1);
            };
            filter2.val("<" + filter_len.slice(filter_len.indexOf('<')+1));
        }
        else {
            //警告框
            $("#filter_EC").after(htmlWarning_E("Word length screening condition '"+filter_len+"' is out of range or unrecognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter2.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.l == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the word length screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_1stPY != "") {
        filter_1stPY = filter_1stPY.toLowerCase()
        var reg = /[a-z]/g;
        if (filter_1stPY.replace(reg, "")=="") { //证明只有英文字母
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
                var pyszm = value.s.split(" ");
                for (var i=0;i<filter_1stPY.length;i++) {
                    if (pyszm[i]!=filter_1stPY[i]) {return false;};
                };
                return true;
            };
            filter3.val(filter_1stPY);
        }
        else {
            //警告框
            $("#filter_EC").after(htmlWarning_E("Initial Pinyin screening condition '"+filter_1stPY+"' is not recognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter3.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the initial Pinyin screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    if (filter_strok != "") {
        if (filter_strok>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b == filter_strok;
            };
        }
        else if (filter_strok.indexOf('>')>-1 && filter_strok.slice(filter_strok.indexOf('>')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b > filter_strok.slice(filter_strok.indexOf('>')+1);
            };
            document.getElementById("filter4_EC").value = ">" + filter_strok.slice(filter_strok.indexOf('>')+1);
        }
        else if (filter_strok.indexOf('<')>-1 && filter_strok.slice(filter_strok.indexOf('<')+1)>0) {
            var dictList_filtered = dictList_filtered.filter(localFunc);
            function localFunc(value) {
              return value.b < filter_strok.slice(filter_strok.indexOf('<')+1);
            };
            document.getElementById("filter4_EC").value = "<" + filter_strok.slice(filter_strok.indexOf('<')+1);
        }
        else {
            //警告框
            $("#filter_EC").after(htmlWarning_E("Number of strokes screening condition '"+filter_strok+"' is not recognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            document.getElementById("filter4_EC").value = "";
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the number of strokes screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    //*为匹配0到多字；？匹配1字；+为且；[...]匹配集合内任一字；[^...]不匹配集合内任何字
    if (filter_shape != "") {
        var reg = /[\u4e00-\u9fa5]/g;
        var ruleStr = "或********或????????或？？？？？？？？或++++++++或[^]或[]"; //多次匹配模式（第一个“或”字占位符必须加，因为如果搜索目标是空的则搜索结果是位置0）
        //var ruleStr = "或*或?或？或+或[^]或[]"; //单次匹配模式
        var ruleInd = ruleStr.indexOf(filter_shape.replace(reg, ""));
        var tmp = filter_shape.match(reg);
        try {
            var hanziStr = tmp.join("");
        }
        catch(err) {
            var hanziStr = "";
        };
        if (ruleInd>-1) {
            if (ruleStr[ruleInd]=='*') {
                var hanziArr = filter_shape.split('*');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    var tmp = [];
                    for (var i=0;i<this.length;i++) { // 山*水* --> ["山","水",""]，有一个空，因为*在边上的原因。
                        if (this[i].length>0) {
                            tmp.push(this[i]);
                        };
                    };
                    if (tmp.length==0) { return true;}; //没有汉字，则都算符合。
                    if (this[0]!="") { // 开头不是*而是字时，必须匹配第一个字/词
                        if (value.w[0]!=this[0]) {return false;};
                    };
                    if (this[this.length-1]!="") { // 结尾不是*而是字时，必须匹配最后一个字/词
                        if (value.w[value.w.length-1]!=this[this.length-1]) {return false;};
                    };
                    if (tmp.length==1) { //一个字或词，找到就符合。
                        if (value.w.indexOf(tmp[0])>-1) {
                            return true;
                        }
                        else {
                            return false;
                        };
                    }
                    else {
                        var ind = value.w.indexOf(tmp[0]);
                        if (ind<0) {return false;};
                        for (var i=1;i<tmp.length;i++) { //多个字或词，从上一次找到的点往后找，以保证按顺序。
                            if (value.w.indexOf(tmp[i], ind+1)<0) {
                                return false;
                            }
                            else {
                                ind = value.w.indexOf(tmp[i]);
                            };
                        };
                        return true;
                    };
                }, hanziArr);
            }
            else if (ruleStr[ruleInd]=='?' || ruleStr[ruleInd]=='？') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    if (filter_shape.length!=value.w.length) {return false};
                    for (var i=0;i<filter_shape.length;i++) {
                        if (filter_shape[i]==ruleStr[ruleInd]) {continue;}
                        else {
                            if (filter_shape[i]!=value.w[i]) {return false;};
                        };
                    };
                    return true;
                });
            }
            else if (ruleStr[ruleInd]=='+') {
                var hanziArr = filter_shape.split('+');
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])<0) {return false;};
                    };
                    return true;
                }, hanziArr);
            }
            else if (ruleStr[ruleInd]=='[' && ruleStr[ruleInd+1]=='^') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])>-1) {return false;};
                    };
                    return true;
                }, hanziStr);
            }
            else if (ruleStr[ruleInd]=='[') {
                var dictList_filtered = dictList_filtered.filter(function (value) {
                    for (var i=0;i<this.length;i++) {
                        if (value.w.indexOf(this[i])>-1) {return true;};
                    };
                    return false;
                }, hanziStr);
            }
            else {
                //警告框
                $("#filter_EC").after(htmlWarning_E("Wildcard patterns screening condition '"+filter_shape+"' is not recognizable."));
                $(".alert").on("click", function(){$(this).slideUp("fast");});
                filter5.val(this.defaultValue);
                return false;
            };
        }
        else {
            //警告框
            $("#filter_EC").after(htmlWarning_E("Wildcard patterns screening condition '"+filter_shape+"' is not recognizable."));
            $(".alert").on("click", function(){$(this).slideUp("fast");});
            filter5.val(this.defaultValue);
            return false;
        };
    };
    if (dictList_filtered.length == 0) {
        //信息框
        $("#filter_EC").after(htmlInfo_E("No screening results, please modify the Wildcard patterns screening condition."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return false;
    };
    dictList_filtered = dictList_filtered.slice(0,100);
    switch (sort_rule) {
        case 1:
            dictList_filtered.sort(function(a, b){
                if (a.s[0] > b.s[0]) {
                    return 1;
                }
                else if (a.s[0] < b.s[0]) {
                    return -1;
                }
                else {
                    return 0;
                }
            });
            break;
        case 2:
            dictList_filtered.sort(function(a, b){
                if (a.s[0] > b.s[0]) {
                    return -1;
                }
                else if (a.s[0] < b.s[0]) {
                    return 1;
                }
                else {
                    return 0;
                }
            });
            break;
        case 3:
            dictList_filtered.sort(function(a, b){return a.b - b.b});
            break;
        case 4:
            dictList_filtered.sort(function(a, b){return b.b - a.b});
            break;
        case 5:
            dictList_filtered.sort(function(a, b){return a.B - b.B});
            break;
        case 6:
            dictList_filtered.sort(function(a, b){return b.B - a.B});
            break;
    };
    showTable(dictList_filtered, $('#result_EC'));
};


function modelProcecss_EC() {
    clearAlert();
    var description = $("#description_EC").val();
    if (description.length==0) {
        $("#filter_EC").after(htmlDanger_E("The input description cannot be empty."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    var reg = /[a-zA-Z]/;
    if (description.search(reg)<0) {
        $("#filter_EC").after(htmlDanger_E("The input characters are unrecognizable."));
        $(".alert").on("click", function(){$(this).slideUp("fast");});
        return true;
    };
    
    // 聚类功能
    if ($(window).width()<751 || window.innerWidth<768) {
        var main_select=$("#filter_EC div.visible-xs").find("#main_select_EC");
    }
    else {
        var main_select=$("#filter_EC div.visible-lg").find("#main_select_EC"); 
    };
    var sort_rule = main_select[0].selectedIndex;
    if (sort_rule==7) {
        $.get("/ChineseRDCluster/", { 'description': description, 'mode': 'EC' }, function (ret) {
            showTable_Cluster(ret, $('#result_EC'));
        });
        return true;
    }
    $("#filter_EC div").find("*").removeAttr("disabled");
    if ($("#description_EC").val()==description_backup_EC) {
        filterRes_EC(retData_backup_EC);
    }
    else {
        $.get("/ChineseRD/", { 'description': description, 'mode': 'EC' }, function (ret) {
            retData_backup_EC = ret.slice(0);
            description_backup_EC = description.slice(0);
            //console.log(ret);
            try {
                filterRes_EC(retData_backup_EC);
                $("#filter_EC").show();
            }
            catch(err) {
                $('#result_EC').html("");
                switch (ret['error']){
                    case 0: //错误框
                        $("#filter_EC").after(htmlDanger_E("The input description cannot be empty."));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    case 1: //错误框
                        $("#filter_EC").after(htmlDanger_E("The input characters are unrecognizable."));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                        break;
                    default: //报出明确的错误类型。
                        $("#filter_EC").after(htmlDanger(err.name));
                        $(".alert").on("click", function(){$(this).slideUp("fast");});
                }
            }
        });
    }
};
function onkeySearch_EC() {
    $('#result_EC').html("");
    clearAlert();
    modelProcecss_EC();
};
$(document).ready(function () {
    $("#description_EC").keypress(function(e) {
        if(e.keyCode == 13)
            {
                $('#result_EC').html("");
                clearAlert();
                modelProcecss_EC();
            }
    });
});

<!----------------------------英Outlook--------------------------------->
$(document).ready(function () {
    $("#description_EO").keypress(function(e) {
        if(e.keyCode == 13)
            {
                var description = $("#description_EO").val();
                $.get("/EnglishRD/", { 'description': description, 'mode': 'EO' }, function (ret) {
                    $('#result_EO').html(ret)
                    })
            }
    });
});

<!--------筛选和排序--------->
$(document).ready(function(){
  $("#flip_EE").click(function(){
    $("#panel_EE").slideToggle("fast", function(){
    if($(this).is(":visible")){
        $("#flip_EE").html('Clear and Hide Filter <span class="glyphicon glyphicon-off"></span>')} <!--和排序<span class="glyphicon glyphicon-sort"></span>-->
    else{
        clearAlert();
        $("#filter_EE div").find("*").removeAttr("disabled");
        try {
            if ($("#description_EE").val()=="") {
                $('#result_EE').html("");
            }
            else {
                if ($("#description_EE").val()==description_backup_EE) {
                    showTable(retData_backup_EE, $('#result_EE'));
                }
                else {
                    modelProcecss_EE();
                };
            };
        }
        catch(err) {
            $('#result_EE').html("");
        };                            
        $("#filter_EE.panel").find("*").val(this.defaultValue).css("background-color", "");
        $("#flip_EE").html('Open Filter <span class="glyphicon glyphicon-filter"></span>')}
        document.getElementById("main_select_EE").options.selectedIndex = 0;
        document.getElementById("POS_select_EE").options.selectedIndex = 0;
    });
  });
});
$(document).ready(function(){
  $("#flip_CE").click(function(){
    $("#panel_CE").slideToggle("fast", function(){
    if($(this).is(":visible")){
        $("#flip_CE").html('清除并收起 筛选器 <span class="glyphicon glyphicon-off"></span>')} <!--和排序<span class="glyphicon glyphicon-sort"></span>-->
    else{
        clearAlert();
        $("#filter_CE div").find("*").removeAttr("disabled");
        try {
            if ($("#description_CE").val()=="") {
                $('#result_CE').html("");
            }
            else {
                if ($("#description_CE").val()==description_backup_CE) {
                    showTable(retData_backup_CE, $('#result_CE'));
                }
                else {
                    modelProcecss_CE();
                };
            };
        }
        catch(err) {
            $('#result_CE').html("");
        };                            
        $("#filter_CE.panel").find("*").val(this.defaultValue).css("background-color", "");
        $("#flip_CE").html('开启 筛选器 <span class="glyphicon glyphicon-filter"></span>')}
        document.getElementById("main_select_CE").options.selectedIndex = 0;
        document.getElementById("POS_select_CE").options.selectedIndex = 0;
    });
  });
});
$(document).ready(function(){
  $("#flip_EC").click(function(){
    $("#panel_EC").slideToggle("fast", function(){
    if($(this).is(":visible")){
        $("#flip_EC").html('Clear and Hide Filter <span class="glyphicon glyphicon-off"></span>')} 
    else{
        clearAlert();
        $("#filter_EC div").find("*").removeAttr("disabled");
        try {
            if ($("#description_EC").val()=="") {
                $('#result_EC').html("");
            }
            else {
                if ($("#description_EC").val()==description_backup_EC) {
                    showTable(retData_backup_EC, $('#result_EC'));
                }
                else {
                    modelProcecss_EC();
                };
            };
        }
        catch(err) {
            $('#result_EC').html("");
        };                            
        $("#filter_EC div.panel").find("*").val(this.defaultValue).css("background-color", "");
        $("#flip_EC").html('Open Filter <span class="glyphicon glyphicon-filter"></span>')}
        document.getElementById("main_select_EC").options.selectedIndex = 0;
        document.getElementById("POS_select_EC").options.selectedIndex = 0;
        document.getElementById("rhyme_select_EC").options.selectedIndex = 0;
    });
  });
});
$(document).ready(function(){
  $("#flip").click(function(){
    $("#panel").slideToggle("fast", function(){
    if($(this).is(":visible")){
        $("#flip").html('清除并收起 筛选器 <span class="glyphicon glyphicon-off"></span>')} <!--和排序<span class="glyphicon glyphicon-sort"></span>-->
    else{
        clearAlert();
        $("#filter_CN div").find("*").removeAttr("disabled");
        try {
            if ($("#description").val()=="") { //收起筛选器后，若输入为空（可能一开始就是空，或改为空但没按回车）则清空 输出区。
                $('#result').html("");
            }
            else {
                if ($("#description").val()==description_backup) { //输入框中没变化，则因为没有筛选条件而直接显示上一次的结果。
                    showTable(retData_backup, $('#result'));
                }
                else {
                    modelProcecss(); //输入框里有变化，则重新计算结果（没有筛选条件，filterRes中的判断都会跳过的，不慢）。
                };
            };
        }
        catch(err) {
            $('#result').html("");
        };                            
        $("#filter_CN div.panel").find("*").val(this.defaultValue);
        $("#flip").html('开启 筛选器 <span class="glyphicon glyphicon-filter"></span>')}
        document.getElementById("main_select").options.selectedIndex = 0;
        document.getElementById("POS_select_CC").options.selectedIndex = 0;
        document.getElementById("rhyme_select_CC").options.selectedIndex = 0;                        
    });
  });
});