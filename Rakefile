require "date"

namespace :appcache do
  desc "update the date in the appcache file (in the gh-pages branch)"
  task :update do
    appcache = File.read("cache.appcache")
    appcacheb = File.read("cacheb.appcache")
    updated  = "# Updated: #{DateTime.now}"

    File.write("cache.appcache", appcache.sub(/^# Updated:.*$/, updated))
    File.write("cacheb.appcache", appcacheb.sub(/^# Updated:.*$/, updated))
  end
end
